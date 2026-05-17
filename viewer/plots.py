"""pyqtgraph-based plot widgets for the ROGII viewer."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from viewer.data import (
    FORMATION_COLORS,
    FORMATION_TOPS,
    Predictions,
    WellBundle,
    predictions_aligned_to_hw,
    rmse,
)
from viewer.heatmap import HAS_DTW, HeatmapBundle, compute_gr_heatmap, map_path_to_tvt

pg.setConfigOptions(antialias=True, background="w", foreground="#1f2933")


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ---------------------------------------------------------------------------
# Cross-section: Z vs MD with formation-top bands and PS marker
# ---------------------------------------------------------------------------
class CrossSectionWidget(QtWidgets.QWidget):
    """Geological cross-section along the wellbore.

    X axis: MD (ft). Y axis: elevation (ft, negative = deeper).
    Wellbore Z is overlaid on top of the colored formation bands.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "MD", units="ft")
        self.plot.setLabel("left", "Elevation Z", units="ft")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))
        layout.addWidget(self.plot)

        self._ps_line: Optional[pg.InfiniteLine] = None

    def clear(self) -> None:
        self.plot.clear()
        self._ps_line = None

    def show_well(self, w: WellBundle) -> None:
        self.clear()
        plot = self.plot

        md = w.md
        z = w.hw["Z"].to_numpy()

        # 1) Formation-top color bands (training-only). Drawn first so the wellbore lands on top.
        if w.has_formation_tops:
            # Each row contains the elevation of each formation top at that MD.
            # The bands stretch between consecutive formation tops (shallow -> deep).
            tops = [w.hw[c].to_numpy() for c in FORMATION_TOPS]
            top_above_ancc = tops[0] + 80.0  # synthetic "above-ANCC" cap, 80 ft thick
            top_below_buda = tops[-1] - 80.0  # synthetic "below-BUDA" floor

            layers = []
            # Layer above ANCC (light grey, just for visual context)
            layers.append(("above ANCC", top_above_ancc, tops[0], "#e8eaed"))
            for i in range(len(FORMATION_TOPS) - 1):
                name = FORMATION_TOPS[i]
                layers.append((name, tops[i], tops[i + 1], FORMATION_COLORS[name]))
            layers.append((FORMATION_TOPS[-1], tops[-1], top_below_buda, FORMATION_COLORS[FORMATION_TOPS[-1]]))

            for name, upper, lower, color in layers:
                # Fill between two curves. pyqtgraph fillbetween needs two PlotDataItems.
                c1 = pg.PlotCurveItem(md, upper, pen=pg.mkPen(_hex_to_rgba(color, 150), width=0))
                c2 = pg.PlotCurveItem(md, lower, pen=pg.mkPen(_hex_to_rgba(color, 150), width=0))
                fill = pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(_hex_to_rgba(color, 180)))
                plot.addItem(c1)
                plot.addItem(c2)
                plot.addItem(fill)
                # Top boundary line (formation top itself)
                if name in FORMATION_COLORS:
                    plot.plot(
                        md,
                        upper,
                        pen=pg.mkPen(_hex_to_rgba(FORMATION_COLORS[name], 220), width=1.4),
                        name=name,
                    )

        # 2) Wellbore Z trajectory
        plot.plot(md, z, pen=pg.mkPen("#1f2933", width=2.4), name="Wellbore Z")

        # 3) PS marker
        if w.ps_md is not None:
            ps_line = pg.InfiniteLine(
                pos=w.ps_md,
                angle=90,
                pen=pg.mkPen("#d05a5a", width=1.4, style=QtCore.Qt.DashLine),
                label="Prediction Start",
                labelOpts={"position": 0.92, "color": "#d05a5a", "fill": (255, 255, 255, 220)},
            )
            plot.addItem(ps_line)
            self._ps_line = ps_line

        plot.enableAutoRange()


# ---------------------------------------------------------------------------
# GR correlation: TVT (y, inverted) vs GR (x)
# ---------------------------------------------------------------------------
class GRCorrelationWidget(QtWidgets.QWidget):
    """Two stacked panels showing GR-to-TVT correlation.

    Top    : Horizontal-well GR vs MD with PS marker.
    Bottom : TVT (y, inverted) vs GR (x).
             Typewell GR as a black line, known HW GR as green points,
             optional predicted (TVT, GR) as red points.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.gr_md_plot = pg.PlotWidget()
        self.gr_md_plot.setLabel("bottom", "MD", units="ft")
        self.gr_md_plot.setLabel("left", "GR")
        self.gr_md_plot.showGrid(x=True, y=True, alpha=0.25)
        self.gr_md_plot.setMinimumHeight(140)

        self.gr_tvt_plot = pg.PlotWidget()
        self.gr_tvt_plot.setLabel("bottom", "GR")
        self.gr_tvt_plot.setLabel("left", "TVT", units="ft")
        self.gr_tvt_plot.showGrid(x=True, y=True, alpha=0.25)
        self.gr_tvt_plot.invertY(True)
        self.gr_tvt_plot.addLegend(offset=(10, 10))

        layout.addWidget(self.gr_md_plot, stretch=1)
        layout.addWidget(self.gr_tvt_plot, stretch=3)

    def clear(self) -> None:
        self.gr_md_plot.clear()
        self.gr_tvt_plot.clear()

    def show_well(
        self,
        w: WellBundle,
        pred_for_well: Optional[pd.DataFrame] = None,
    ) -> None:
        self.clear()

        md = w.md
        gr_hw = w.hw["GR"].to_numpy() if "GR" in w.hw.columns else np.full(len(md), np.nan)

        # Top: GR vs MD
        finite = np.isfinite(gr_hw)
        if finite.any():
            self.gr_md_plot.plot(
                md[finite],
                gr_hw[finite],
                pen=pg.mkPen("#3aa856", width=1.1),
            )
        if w.ps_md is not None:
            self.gr_md_plot.addItem(
                pg.InfiniteLine(
                    pos=w.ps_md,
                    angle=90,
                    pen=pg.mkPen("#d05a5a", width=1.2, style=QtCore.Qt.DashLine),
                )
            )

        # Bottom: TVT vs GR
        tw_gr = w.tw["GR"].to_numpy() if "GR" in w.tw.columns else None
        tw_tvt = w.tw["TVT"].to_numpy() if "TVT" in w.tw.columns else None
        if tw_gr is not None and tw_tvt is not None:
            tw_mask = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
            self.gr_tvt_plot.plot(
                tw_gr[tw_mask],
                tw_tvt[tw_mask],
                pen=pg.mkPen("#1f2933", width=1.4),
                name="typewell GR",
            )

        # HW known points (GR vs TVT_input)
        tvt_input = w.hw["TVT_input"].to_numpy() if "TVT_input" in w.hw.columns else None
        if tvt_input is not None:
            mask = np.isfinite(tvt_input) & np.isfinite(gr_hw)
            if mask.any():
                self.gr_tvt_plot.plot(
                    gr_hw[mask],
                    tvt_input[mask],
                    pen=None,
                    symbol="o",
                    symbolSize=4,
                    symbolPen=None,
                    symbolBrush=(58, 168, 86, 110),
                    name="HW known",
                )

        # HW truth points (training only, post-PS) — gray for context
        if w.has_truth and w.ps_idx is not None:
            tvt = w.hw["TVT"].to_numpy()
            post = slice(w.ps_idx, None)
            mask = np.isfinite(tvt[post]) & np.isfinite(gr_hw[post])
            if mask.any():
                self.gr_tvt_plot.plot(
                    gr_hw[post][mask],
                    tvt[post][mask],
                    pen=None,
                    symbol="o",
                    symbolSize=4,
                    symbolPen=None,
                    symbolBrush=(120, 120, 120, 100),
                    name="HW truth (post-PS)",
                )

        # Predictions overlay
        if pred_for_well is not None and w.ps_idx is not None:
            md_p, tvt_p, _ = predictions_aligned_to_hw(pred_for_well, w.hw)
            # Aligned GR for the predicted rows
            idx = pred_for_well["row_index"].to_numpy()
            valid = (idx >= 0) & (idx < len(md))
            idx = idx[valid]
            gr_p = gr_hw[idx]
            mask = np.isfinite(gr_p) & np.isfinite(tvt_p)
            if mask.any():
                self.gr_tvt_plot.plot(
                    gr_p[mask],
                    tvt_p[mask],
                    pen=None,
                    symbol="x",
                    symbolSize=6,
                    symbolPen=pg.mkPen("#d05a5a", width=1.5),
                    symbolBrush=None,
                    name="prediction",
                )


# ---------------------------------------------------------------------------
# Map view: X/Y trajectories
# ---------------------------------------------------------------------------
class MapWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "X", units="ft")
        self.plot.setLabel("left", "Y", units="ft")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.getViewBox().setAspectLocked(True, 1.0)
        layout.addWidget(self.plot)

        self._background_plotted = False
        self._current_items: list = []

    def plot_background(self, centroids: list[tuple[float, float, str]]) -> None:
        """One-shot light-gray scatter of all well centroids."""
        if self._background_plotted or not centroids:
            return
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        self.plot.plot(
            xs,
            ys,
            pen=None,
            symbol="o",
            symbolSize=3,
            symbolPen=None,
            symbolBrush=(170, 170, 170, 130),
        )
        self._background_plotted = True

    def show_well(self, w: WellBundle, neighbours: Optional[list[tuple[float, float, str]]] = None) -> None:
        for item in self._current_items:
            self.plot.removeItem(item)
        self._current_items.clear()

        if neighbours:
            xs = [n[0] for n in neighbours]
            ys = [n[1] for n in neighbours]
            item = self.plot.plot(
                xs,
                ys,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolPen=pg.mkPen("#3b6fbb", width=1.2),
                symbolBrush=(59, 111, 187, 160),
            )
            self._current_items.append(item)

        x = w.hw["X"].to_numpy()
        y = w.hw["Y"].to_numpy()
        item = self.plot.plot(x, y, pen=pg.mkPen("#d05a5a", width=2.4))
        self._current_items.append(item)
        # Toe marker
        toe = self.plot.plot(
            [x[-1]],
            [y[-1]],
            pen=None,
            symbol="t",
            symbolSize=12,
            symbolPen=pg.mkPen("#d05a5a", width=1),
            symbolBrush=(208, 90, 90, 220),
        )
        self._current_items.append(toe)


# ---------------------------------------------------------------------------
# TVT prediction vs truth panel (only useful when truth is available)
# ---------------------------------------------------------------------------
class TVTPredictionWidget(QtWidgets.QWidget):
    """Plot TVT vs MD with TVT_input, truth (if available), and predictions overlaid."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "MD", units="ft")
        self.plot.setLabel("left", "TVT", units="ft")
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))
        layout.addWidget(self.plot)

    def show_well(
        self,
        w: WellBundle,
        pred_for_well: Optional[pd.DataFrame] = None,
    ) -> tuple[Optional[float], int]:
        """Returns (rmse_against_truth_or_None, n_predicted_points)."""
        self.plot.clear()
        md = w.md

        if w.has_truth:
            self.plot.plot(
                md,
                w.hw["TVT"].to_numpy(),
                pen=pg.mkPen("#3aa856", width=1.6),
                name="TVT truth",
            )
        if "TVT_input" in w.hw.columns:
            ti = w.hw["TVT_input"].to_numpy()
            mask = np.isfinite(ti)
            if mask.any():
                self.plot.plot(
                    md[mask],
                    ti[mask],
                    pen=pg.mkPen("#e9924b", width=1.8),
                    name="TVT_input (known)",
                )

        if w.ps_md is not None:
            self.plot.addItem(
                pg.InfiniteLine(
                    pos=w.ps_md,
                    angle=90,
                    pen=pg.mkPen("#d05a5a", width=1.4, style=QtCore.Qt.DashLine),
                )
            )

        rmse_val: Optional[float] = None
        n_pred = 0
        if pred_for_well is not None:
            md_p, tvt_p, truth = predictions_aligned_to_hw(pred_for_well, w.hw)
            if len(md_p) > 0:
                self.plot.plot(
                    md_p,
                    tvt_p,
                    pen=pg.mkPen("#d05a5a", width=1.6, style=QtCore.Qt.DotLine),
                    name="prediction",
                )
                n_pred = len(md_p)
                if w.has_truth and np.isfinite(truth).any():
                    rmse_val = rmse(tvt_p, truth)
        return rmse_val, n_pred


# ---------------------------------------------------------------------------
# GR mismatch heatmap (hengck23-style) + DTW / ground-truth path overlays
# ---------------------------------------------------------------------------
class GRHeatmapWidget(QtWidgets.QWidget):
    """Heatmap-based DTW alignment view of HW GR against the typewell GR.

    Top:    `heatmap[i, j] = HW_seg_GR[j] - typewell_seg_GR[i]` with two path
            overlays — yellow = ground-truth typewell-row index per HW segment
            (only when TVT is available), orange dashed = DTW-computed path.
    Below:  three small panels — HW GR per segment (colored by index), typewell
            GR vs TVT, and the GR profile in TVT coordinates (typewell GR with
            the HW segments projected back via the DTW path).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Controls row
        ctl = QtWidgets.QHBoxLayout()
        ctl.setContentsMargins(6, 4, 6, 0)
        ctl.addWidget(QtWidgets.QLabel("Segment size S (ft):"))
        self.s_spin = QtWidgets.QSpinBox()
        self.s_spin.setRange(4, 128)
        self.s_spin.setValue(32)
        self.s_spin.setSingleStep(4)
        ctl.addWidget(self.s_spin)
        ctl.addSpacing(12)
        ctl.addWidget(QtWidgets.QLabel("Before PS:"))
        self.before_spin = QtWidgets.QSpinBox()
        self.before_spin.setRange(0, 64)
        self.before_spin.setValue(8)
        ctl.addWidget(self.before_spin)
        ctl.addWidget(QtWidgets.QLabel("After PS:"))
        self.after_spin = QtWidgets.QSpinBox()
        self.after_spin.setRange(1, 256)
        self.after_spin.setValue(16)
        ctl.addWidget(self.after_spin)
        ctl.addSpacing(12)
        self.show_gt = QtWidgets.QCheckBox("Ground truth")
        self.show_gt.setChecked(True)
        ctl.addWidget(self.show_gt)
        self.show_dtw = QtWidgets.QCheckBox("DTW path")
        self.show_dtw.setChecked(HAS_DTW)
        self.show_dtw.setEnabled(HAS_DTW)
        if not HAS_DTW:
            self.show_dtw.setToolTip("dtw-python is not installed (pip install dtw-python)")
        ctl.addWidget(self.show_dtw)
        ctl.addStretch(1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        ctl.addWidget(self.refresh_btn)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #6b7280; font-size: 11px;")
        ctl.addWidget(self.status)
        root.addLayout(ctl)

        # Plot stack
        self.glw = pg.GraphicsLayoutWidget()
        root.addWidget(self.glw, stretch=1)

        # Top row: big heatmap (full width)
        self.heatmap_plot: pg.PlotItem = self.glw.addPlot(row=0, col=0, colspan=3)
        self.heatmap_plot.setLabel("bottom", "HW segment index (centered at PS)")
        self.heatmap_plot.setLabel("left", "Typewell row index")
        self.heatmap_plot.invertY(True)
        self.heatmap_plot.setMenuEnabled(False)
        self.heatmap_image = pg.ImageItem(axisOrder="row-major")
        self.heatmap_plot.addItem(self.heatmap_image)
        # Diverging colormap — like hengck23's "seismic"/"bwr"
        cmap = pg.colormap.get("CET-D1A") or pg.colormap.get("seismic")
        self.heatmap_lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.heatmap_image.setLookupTable(self.heatmap_lut)
        # Colorbar
        self.cbar = pg.ColorBarItem(values=(-40, 40), colorMap=cmap, label="GR difference",
                                    interactive=False)
        self.cbar.setImageItem(self.heatmap_image, insert_in=self.heatmap_plot)

        # Bottom row: three small panels
        self.hw_plot: pg.PlotItem = self.glw.addPlot(row=1, col=0)
        self.hw_plot.setLabel("bottom", "HW segment index")
        self.hw_plot.setLabel("left", "GR (smoothed, per seg)")
        self.hw_plot.setTitle("Horizontal-well GR per segment")
        self.hw_plot.showGrid(x=True, y=True, alpha=0.25)

        self.tw_plot: pg.PlotItem = self.glw.addPlot(row=1, col=1)
        self.tw_plot.setLabel("bottom", "GR")
        self.tw_plot.setLabel("left", "TVT (ft)")
        self.tw_plot.setTitle("Typewell GR vs TVT")
        self.tw_plot.invertY(True)
        self.tw_plot.showGrid(x=True, y=True, alpha=0.25)

        self.tvt_plot: pg.PlotItem = self.glw.addPlot(row=1, col=2)
        self.tvt_plot.setLabel("bottom", "TVT (ft)")
        self.tvt_plot.setLabel("left", "GR")
        self.tvt_plot.setTitle("HW segments projected onto TVT (DTW)")
        self.tvt_plot.showGrid(x=True, y=True, alpha=0.25)

        self._current: Optional[WellBundle] = None

        # Wire interactions
        self.refresh_btn.clicked.connect(self._refresh)
        self.show_gt.toggled.connect(self._refresh)
        self.show_dtw.toggled.connect(self._refresh)
        self.s_spin.editingFinished.connect(self._refresh)
        self.before_spin.editingFinished.connect(self._refresh)
        self.after_spin.editingFinished.connect(self._refresh)

    # ------------------------------------------------------------------
    def show_well(self, w: WellBundle) -> None:
        self._current = w
        self._refresh()

    def _refresh(self) -> None:
        w = self._current
        if w is None or w.ps_idx is None:
            self.heatmap_plot.clear()
            self.heatmap_plot.addItem(self.heatmap_image)
            self.heatmap_image.setImage(np.zeros((1, 1), dtype=np.float32))
            self.hw_plot.clear()
            self.tw_plot.clear()
            self.tvt_plot.clear()
            self.status.setText("(no well loaded)")
            return

        try:
            bundle = compute_gr_heatmap(
                w.hw, w.tw, w.ps_idx,
                S=int(self.s_spin.value()),
                n_before=int(self.before_spin.value()),
                n_after=int(self.after_spin.value()),
                run_dtw=bool(self.show_dtw.isChecked()),
            )
        except Exception as exc:
            self.heatmap_plot.clear()
            self.heatmap_plot.addItem(self.heatmap_image)
            self.heatmap_image.setImage(np.zeros((1, 1), dtype=np.float32))
            self.hw_plot.clear()
            self.tw_plot.clear()
            self.tvt_plot.clear()
            self.status.setText(f"compute failed: {exc}")
            return

        # Heatmap image: extent = [x0, y0, x_span, y_span]; we want x = HW seg
        # index (0..n_hw), y = typewell row index (0..n_tw).
        self.heatmap_image.setImage(
            bundle.heatmap,
            autoLevels=False,
            levels=(-40.0, 40.0),
        )
        # Position the image so (x, y) = (seg_index, row_index) with 1:1 cells
        self.heatmap_image.setRect(QtCore.QRectF(0, 0, bundle.n_hw, bundle.n_tw))
        # Reset overlay items (paths + PS line)
        for item in list(self.heatmap_plot.items):
            if isinstance(item, (pg.PlotDataItem, pg.InfiniteLine)) and item is not self.heatmap_image:
                self.heatmap_plot.removeItem(item)

        # PS marker
        ps_line = pg.InfiniteLine(
            pos=bundle.ps_seg_index + 0.5, angle=90,
            pen=pg.mkPen("#1f2933", width=1.2, style=QtCore.Qt.DashLine),
            label=f"PS (seg {bundle.ps_seg_index})",
            labelOpts={"position": 0.05, "color": "#1f2933"},
        )
        self.heatmap_plot.addItem(ps_line)

        # Ground-truth path (yellow)
        if self.show_gt.isChecked() and bundle.gt_path is not None:
            xs = np.arange(bundle.n_hw) + 0.5
            ys = bundle.gt_path.astype(float) + 0.5
            self.heatmap_plot.plot(xs, ys,
                                   pen=pg.mkPen("#000000", width=4),
                                   name="GT")
            self.heatmap_plot.plot(xs, ys,
                                   pen=pg.mkPen("#f4d03f", width=2),
                                   name="GT")
        # DTW path (orange dashed)
        if self.show_dtw.isChecked() and bundle.dtw_path is not None:
            xs = np.arange(bundle.n_hw) + 0.5
            ys = bundle.dtw_path.astype(float) + 0.5
            self.heatmap_plot.plot(xs, ys,
                                   pen=pg.mkPen("#e9924b", width=2.0,
                                                style=QtCore.Qt.DashLine),
                                   name="DTW")

        # HW GR panel
        self.hw_plot.clear()
        seg_idx = np.arange(bundle.n_hw)
        self.hw_plot.plot(seg_idx, bundle.h_seg_gr,
                          pen=pg.mkPen("#e9924b", width=1.5))
        self.hw_plot.plot(seg_idx, bundle.h_seg_gr,
                          pen=None, symbol="o", symbolSize=6,
                          symbolPen=None,
                          symbolBrush=[pg.intColor(i, hues=max(bundle.n_hw, 6))
                                       for i in range(bundle.n_hw)])
        self.hw_plot.addItem(pg.InfiniteLine(pos=bundle.ps_seg_index, angle=90,
                                             pen=pg.mkPen("#d05a5a", width=1.2,
                                                          style=QtCore.Qt.DashLine)))

        # TW GR panel — GR (x) vs TVT (y, inverted)
        self.tw_plot.clear()
        self.tw_plot.plot(bundle.t_seg_gr, bundle.t_seg_tvt,
                          pen=pg.mkPen("#1f2933", width=1.4))

        # GR profile in TVT coordinates: typewell line + HW segments at DTW TVT
        self.tvt_plot.clear()
        self.tvt_plot.plot(bundle.t_seg_tvt, bundle.t_seg_gr,
                           pen=pg.mkPen("#1f2933", width=1.3), name="typewell")
        if bundle.dtw_path is not None and self.show_dtw.isChecked():
            tvt_hw = map_path_to_tvt(bundle.dtw_path, bundle.t_seg_tvt)
            self.tvt_plot.plot(tvt_hw, bundle.h_seg_gr,
                               pen=pg.mkPen("#e9924b", width=1.2),
                               symbol="o", symbolSize=6,
                               symbolPen=None,
                               symbolBrush=[pg.intColor(i, hues=max(bundle.n_hw, 6))
                                            for i in range(bundle.n_hw)],
                               name="HW (DTW projected)")
        elif bundle.gt_path is not None and self.show_gt.isChecked():
            tvt_hw = map_path_to_tvt(bundle.gt_path, bundle.t_seg_tvt)
            self.tvt_plot.plot(tvt_hw, bundle.h_seg_gr,
                               pen=pg.mkPen("#f4d03f", width=1.2),
                               symbol="o", symbolSize=6,
                               symbolPen=None,
                               symbolBrush=[pg.intColor(i, hues=max(bundle.n_hw, 6))
                                            for i in range(bundle.n_hw)],
                               name="HW (GT projected)")

        # Status line: brief alignment quality summary
        parts = [f"S={bundle.S} ft", f"HW={bundle.n_hw} segs", f"TW={bundle.n_tw} rows"]
        if bundle.gt_path is not None and bundle.dtw_path is not None:
            disagreement = int(np.sum(bundle.gt_path != bundle.dtw_path))
            parts.append(f"DTW≠GT on {disagreement}/{bundle.n_hw} segs")
        elif bundle.gt_path is None:
            parts.append("(no GT — test well)")
        if not HAS_DTW:
            parts.append("(install dtw-python for DTW)")
        self.status.setText("   ·   ".join(parts))
