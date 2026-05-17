"""Render all four tabs of the viewer to PNG files under viewer/docs/.

Used to generate screenshots for the README.
Run via:  QT_QPA_PLATFORM=offscreen python -m viewer._screenshots
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets


def main() -> int:
    from viewer.app import MainWindow
    from viewer.data import Predictions

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1400, 820)
    root = Path.home() / "ROGII" / "rogii-wellbore-geology-prediction"
    win.open_dataset(root)

    # Pick a train well with formation tops, then overlay sample_submission
    win._on_well_selected("train/000d7d20")
    win.predictions = Predictions.load(root / "sample_submission.csv")
    win._refresh_current_well()
    QtWidgets.QApplication.processEvents()

    out_dir = Path("/home/tom99763/ROGII/viewer/docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    tabs = {
        "cross_section.png": win.cross_section,
        "tvt_prediction.png": win.tvt_pred,
        "gr_correlation.png": win.gr_corr,
        "gr_heatmap.png": win.gr_heatmap,
        "map_view.png": win.map_view,
    }
    for name, widget in tabs.items():
        widget.setMinimumSize(1100, 620)
        QtWidgets.QApplication.processEvents()
        pix = widget.grab()
        path = out_dir / name
        pix.save(str(path), "PNG")
        print(f"saved {path}  ({path.stat().st_size//1024} KB)")

    # Full window
    QtWidgets.QApplication.processEvents()
    win_pix = win.grab()
    win_pix.save(str(out_dir / "main_window.png"), "PNG")
    print(f"saved {out_dir / 'main_window.png'}  ({(out_dir / 'main_window.png').stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
