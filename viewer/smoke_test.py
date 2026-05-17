"""Offscreen smoke test for the viewer.

Builds the MainWindow, loads the bundled dataset, walks through:
  1. A train well (renders all four tabs).
  2. A test well.
  3. Loads sample_submission.csv as a prediction overlay and re-renders.

Exits with code 0 on success, non-zero on failure.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets


def main() -> int:
    from viewer.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    win = MainWindow()
    root = Path.home() / "ROGII" / "rogii-wellbore-geology-prediction"
    if not root.is_dir():
        print(f"FAIL: dataset not found at {root}", file=sys.stderr)
        return 2
    win.open_dataset(root)
    assert win.dataset is not None
    assert win.dataset.wells, "no wells indexed"

    # First train well
    train_well = next(w for w in win.dataset.wells if w["split"] == "train")
    win._on_well_selected(f"train/{train_well['well_id']}")
    assert win.current_well is not None, "current_well not set after train selection"
    cw = win.current_well
    assert cw.has_truth, "expected TVT column in train well"
    assert cw.has_formation_tops, "expected formation-top columns in train well"
    print(f"OK loaded train well {cw.well_id}: rows={cw.n_rows} PS_idx={cw.ps_idx} has_truth={cw.has_truth} has_tops={cw.has_formation_tops}")

    # First test well
    test_well = next((w for w in win.dataset.wells if w["split"] == "test"), None)
    if test_well is not None:
        win._on_well_selected(f"test/{test_well['well_id']}")
        cw = win.current_well
        assert not cw.has_truth, "test well should not have TVT"
        print(f"OK loaded test well {cw.well_id}: rows={cw.n_rows} PS_idx={cw.ps_idx} has_truth={cw.has_truth}")

    # Load sample_submission.csv as a fake prediction overlay
    from viewer.data import Predictions
    sub_path = root / "sample_submission.csv"
    win.predictions = Predictions.load(sub_path)
    print(f"OK loaded predictions: {win.predictions.coverage()}")
    # Re-render the current well so the overlay shows up
    win._refresh_current_well()
    cw = win.current_well
    print(f"OK refreshed {cw.well_id} with prediction overlay")

    # Now switch to the matching train well (one that overlaps with the submission ids)
    # to verify RMSE computation runs when truth is present.
    pred_well_id = next(iter(win.predictions.by_well))
    matching_train = next((w for w in win.dataset.wells if w["split"] == "train" and w["well_id"] == pred_well_id), None)
    if matching_train is not None:
        win._on_well_selected(f"train/{pred_well_id}")
        cw = win.current_well
        assert cw.split == "train", f"expected train split, got {cw.split}"
        assert cw.has_truth, "expected truth on train well"
        print(f"OK loaded matching train well {pred_well_id} with predictions overlay (RMSE path exercised)")

    # Grab a tab to PNG to make sure rendering actually produced pixels.
    pix = win.cross_section.grab()
    out = Path("/tmp/viewer_cross_section.png")
    pix.save(str(out), "PNG")
    print(f"OK exported cross-section to {out}  ({out.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
