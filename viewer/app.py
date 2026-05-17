"""ROGII Viewer — main window and application entry point."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from viewer import __version__
from viewer.data import (
    DatasetIndex,
    Predictions,
    WellBundle,
    load_well,
    predictions_aligned_to_hw,
    rmse,
)
from viewer.plots import (
    CrossSectionWidget,
    GRCorrelationWidget,
    MapWidget,
    TVTPredictionWidget,
)


# ---------------------------------------------------------------------------
# Left dock: searchable well list
# ---------------------------------------------------------------------------
class WellListPanel(QtWidgets.QWidget):
    well_selected = QtCore.Signal(str)  # emits "split/well_id"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search well id…")
        self.search.textChanged.connect(self._apply_filter)

        self.split_filter = QtWidgets.QComboBox()
        self.split_filter.addItems(["all", "train", "test"])
        self.split_filter.currentTextChanged.connect(self._apply_filter)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.search, 3)
        top.addWidget(self.split_filter, 1)
        layout.addLayout(top)

        self.list = QtWidgets.QListWidget()
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list)

        self.count_label = QtWidgets.QLabel("")
        self.count_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        layout.addWidget(self.count_label)

        self._all_entries: list[dict] = []

    def set_entries(self, entries: list[dict]) -> None:
        self._all_entries = entries
        self._apply_filter()

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        split = self.split_filter.currentText()
        self.list.clear()
        n_shown = 0
        for e in self._all_entries:
            if split != "all" and e["split"] != split:
                continue
            if text and text not in e["well_id"].lower():
                continue
            item = QtWidgets.QListWidgetItem(f"{e['well_id']}    ({e['split']})")
            item.setData(QtCore.Qt.UserRole, f"{e['split']}/{e['well_id']}")
            if e["split"] == "test":
                item.setForeground(QtGui.QBrush(QtGui.QColor("#2b6cb0")))
            self.list.addItem(item)
            n_shown += 1
        self.count_label.setText(f"{n_shown} well(s)")

    def _on_selection_changed(self) -> None:
        items = self.list.selectedItems()
        if not items:
            return
        key = items[0].data(QtCore.Qt.UserRole)
        self.well_selected.emit(key)


# ---------------------------------------------------------------------------
# Right dock: well metadata + prediction info
# ---------------------------------------------------------------------------
class WellInfoPanel(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title = QtWidgets.QLabel("No well loaded")
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(self.title)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

        self.pred_box = QtWidgets.QGroupBox("Predictions")
        pred_layout = QtWidgets.QVBoxLayout(self.pred_box)
        self.pred_file_label = QtWidgets.QLabel("None loaded")
        self.pred_file_label.setWordWrap(True)
        self.pred_file_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        self.pred_rmse_label = QtWidgets.QLabel("RMSE: —")
        self.pred_rmse_label.setStyleSheet("font-weight: 600;")
        self.pred_n_label = QtWidgets.QLabel("Predicted rows: —")
        pred_layout.addWidget(self.pred_file_label)
        pred_layout.addWidget(self.pred_rmse_label)
        pred_layout.addWidget(self.pred_n_label)
        layout.addWidget(self.pred_box)

    def set_metadata(self, rows: list[tuple[str, str]], title: str) -> None:
        self.title.setText(title)
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            it = QtWidgets.QTableWidgetItem(v)
            it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.table.setItem(i, 1, it)
        self.table.resizeColumnsToContents()

    def set_predictions_status(self, filename: Optional[str], coverage: Optional[str]) -> None:
        if filename is None:
            self.pred_file_label.setText("None loaded")
        else:
            self.pred_file_label.setText(f"{filename}  ({coverage})")

    def set_predictions_for_current(self, rmse_val: Optional[float], n_pred: int) -> None:
        if n_pred == 0:
            self.pred_rmse_label.setText("RMSE: —  (no prediction rows for this well)")
            self.pred_n_label.setText("Predicted rows: 0")
            return
        self.pred_n_label.setText(f"Predicted rows: {n_pred:,}")
        if rmse_val is None:
            self.pred_rmse_label.setText("RMSE: —  (no truth available)")
        else:
            self.pred_rmse_label.setText(f"RMSE vs truth: {rmse_val:,.2f} ft")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ROGII Viewer  v{__version__}")
        self.resize(1500, 900)

        self.dataset: Optional[DatasetIndex] = None
        self.current_well: Optional[WellBundle] = None
        self.predictions: Optional[Predictions] = None

        # Central area: three plot tabs in a splitter layout
        self.cross_section = CrossSectionWidget()
        self.gr_corr = GRCorrelationWidget()
        self.tvt_pred = TVTPredictionWidget()
        self.map_view = MapWidget()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.cross_section, "Cross-Section (Z vs MD)")
        self.tabs.addTab(self.tvt_pred, "TVT Prediction vs MD")
        self.tabs.addTab(self.gr_corr, "GR Correlation")
        self.tabs.addTab(self.map_view, "Map View")
        self.setCentralWidget(self.tabs)

        # Left dock: well list
        self.well_list_panel = WellListPanel()
        self.well_list_panel.well_selected.connect(self._on_well_selected)
        left_dock = QtWidgets.QDockWidget("Wells", self)
        left_dock.setWidget(self.well_list_panel)
        left_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, left_dock)

        # Right dock: well info
        self.info_panel = WellInfoPanel()
        right_dock = QtWidgets.QDockWidget("Well Info", self)
        right_dock.setWidget(self.info_panel)
        right_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right_dock)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Open a dataset folder to begin   (File → Open dataset folder)")

        # Menus
        self._build_menus()

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        act_open = QtGui.QAction("Open dataset folder…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._action_open_dataset)
        file_menu.addAction(act_open)

        act_load = QtGui.QAction("Load predictions CSV…", self)
        act_load.setShortcut("Ctrl+L")
        act_load.triggered.connect(self._action_load_predictions)
        file_menu.addAction(act_load)

        act_clear_pred = QtGui.QAction("Clear predictions", self)
        act_clear_pred.triggered.connect(self._action_clear_predictions)
        file_menu.addAction(act_clear_pred)

        file_menu.addSeparator()
        act_export = QtGui.QAction("Export current plot to PNG…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._action_export_png)
        file_menu.addAction(act_export)

        file_menu.addSeparator()
        act_quit = QtGui.QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = bar.addMenu("&Help")
        act_about = QtGui.QAction("About", self)
        act_about.triggered.connect(self._action_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _action_open_dataset(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick the rogii-wellbore-geology-prediction folder"
        )
        if not folder:
            return
        self.open_dataset(Path(folder))

    def open_dataset(self, root: Path) -> None:
        idx = DatasetIndex.scan(root)
        if not idx.wells:
            QtWidgets.QMessageBox.warning(
                self,
                "Empty dataset",
                f"No wells found under {root}.\n\nExpected train/ and/or test/ subfolders with "
                "*__horizontal_well.csv and *__typewell.csv pairs.",
            )
            return
        self.dataset = idx
        self.well_list_panel.set_entries(idx.wells)
        n_train = sum(1 for w in idx.wells if w["split"] == "train")
        n_test = sum(1 for w in idx.wells if w["split"] == "test")
        self.status.showMessage(f"Loaded dataset: {n_train} train + {n_test} test wells from {root}")
        # Plot map background once
        centroids = []
        for entry in idx.wells:
            try:
                import pandas as pd
                df = pd.read_csv(entry["hw"], usecols=["X", "Y"])
                centroids.append((float(df["X"].mean()), float(df["Y"].mean()), entry["well_id"]))
            except Exception:
                continue
        self.map_view.plot_background(centroids)
        self._centroids = centroids

    def _action_load_predictions(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick a Kaggle-format predictions CSV (id, tvt)", filter="CSV (*.csv)"
        )
        if not path:
            return
        try:
            self.predictions = Predictions.load(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load predictions", str(exc))
            return
        self.info_panel.set_predictions_status(
            Path(path).name, self.predictions.coverage()
        )
        self.status.showMessage(f"Loaded predictions: {self.predictions.coverage()}")
        if self.current_well is not None:
            self._refresh_current_well()

    def _action_clear_predictions(self) -> None:
        self.predictions = None
        self.info_panel.set_predictions_status(None, None)
        self.info_panel.set_predictions_for_current(None, 0)
        if self.current_well is not None:
            self._refresh_current_well()

    def _action_export_png(self) -> None:
        widget = self.tabs.currentWidget()
        if widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot as PNG", filter="PNG (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        pix = widget.grab()
        pix.save(path, "PNG")
        self.status.showMessage(f"Saved {path}")

    def _action_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "ROGII Viewer",
            f"<b>ROGII Viewer v{__version__}</b><br><br>"
            "A lightweight desktop geosteering viewer for the<br>"
            "<i>rogii-wellbore-geology-prediction</i> Kaggle dataset.<br><br>"
            "Built with PySide6 + pyqtgraph.",
        )

    # ------------------------------------------------------------------
    # Well selection
    # ------------------------------------------------------------------
    def _on_well_selected(self, key: str) -> None:
        if self.dataset is None:
            return
        entry = self.dataset.find_key(key)
        if entry is None:
            return
        try:
            self.current_well = load_well(entry)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load well", str(exc))
            return
        self._refresh_current_well()

    def _refresh_current_well(self) -> None:
        w = self.current_well
        if w is None:
            return

        # Metadata
        rows = [
            ("Well ID", w.well_id),
            ("Split", w.split),
            ("Rows", f"{w.n_rows:,}"),
            ("MD range", f"{w.hw['MD'].min():,.0f} – {w.hw['MD'].max():,.0f}"),
            ("Z range", f"{w.hw['Z'].min():,.1f} – {w.hw['Z'].max():,.1f}"),
            ("GR missing", f"{w.hw['GR'].isna().mean()*100:.1f}%"),
            ("PS index", str(w.ps_idx) if w.ps_idx is not None else "—"),
            ("PS MD", f"{w.ps_md:,.0f}" if w.ps_md is not None else "—"),
            ("Eval rows", f"{int(w.hw['TVT_input'].isna().sum()):,}"),
            ("Typewell rows", f"{len(w.tw):,}"),
        ]
        if w.has_truth:
            rows.append(("TVT range", f"{w.hw['TVT'].min():,.1f} – {w.hw['TVT'].max():,.1f}"))
        self.info_panel.set_metadata(rows, title=f"{w.well_id}  ({w.split})")

        # Predictions for this well
        pred_for_well = None
        if self.predictions is not None:
            pred_for_well = self.predictions.for_well(w.well_id)

        # Render all plots
        self.cross_section.show_well(w)
        self.gr_corr.show_well(w, pred_for_well=pred_for_well)
        rmse_val, n_pred = self.tvt_pred.show_well(w, pred_for_well=pred_for_well)
        # Neighbours for map
        neighbours = self._nearest_neighbours(w, k=8)
        self.map_view.show_well(w, neighbours=neighbours)
        self.info_panel.set_predictions_for_current(rmse_val, n_pred)

        self.status.showMessage(f"Loaded {w.well_id}  ({w.split})  —  {w.n_rows:,} rows")

    def _nearest_neighbours(self, w: WellBundle, k: int) -> list[tuple[float, float, str]]:
        if not hasattr(self, "_centroids"):
            return []
        cx, cy = w.xy_centroid()
        scored = []
        for x, y, wid in self._centroids:
            if wid == w.well_id:
                continue
            d = (x - cx) ** 2 + (y - cy) ** 2
            scored.append((d, x, y, wid))
        scored.sort()
        return [(x, y, wid) for _, x, y, wid in scored[:k]]


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow()

    # Auto-open the dataset if the conventional path exists, for convenience.
    default_root = Path.home() / "ROGII" / "rogii-wellbore-geology-prediction"
    if default_root.is_dir():
        win.open_dataset(default_root)

    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
