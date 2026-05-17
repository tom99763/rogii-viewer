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
