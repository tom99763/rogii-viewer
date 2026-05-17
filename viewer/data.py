"""Data loading and caching for the ROGII viewer."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# Formation top columns expected only in training horizontal CSVs, ordered shallow -> deep.
FORMATION_TOPS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

# Stable hex colors for the 6 formation bands (shallow -> deep).
FORMATION_COLORS: dict[str, str] = {
    "ANCC": "#c4d3e6",
    "ASTNU": "#9eb9d6",
    "ASTNL": "#7aa1c5",
    "EGFDU": "#dfb88a",  # the two Eagle Ford bands are warmer (productive zone)
    "EGFDL": "#c8965d",
    "BUDA": "#a17a55",
}


@dataclass
class WellBundle:
    """A loaded (horizontal_well, typewell) pair plus derived metadata."""

    well_id: str
    split: str  # "train" or "test"
    hw: pd.DataFrame
    tw: pd.DataFrame
    hw_path: Path
    tw_path: Path

    # Cached scalars
    ps_idx: Optional[int] = None
    ps_md: Optional[float] = None

    def __post_init__(self) -> None:
        if "TVT_input" in self.hw.columns:
            mask = self.hw["TVT_input"].isna().to_numpy()
            if mask.any():
                self.ps_idx = int(np.flatnonzero(mask)[0])
                self.ps_md = float(self.hw["MD"].iloc[self.ps_idx])

    @property
    def has_truth(self) -> bool:
        return "TVT" in self.hw.columns

    @property
    def has_formation_tops(self) -> bool:
        return all(c in self.hw.columns for c in FORMATION_TOPS)

    @property
    def md(self) -> np.ndarray:
        return self.hw["MD"].to_numpy()

    @property
    def n_rows(self) -> int:
        return len(self.hw)

    def xy_centroid(self) -> tuple[float, float]:
        return float(self.hw["X"].mean()), float(self.hw["Y"].mean())


@dataclass
class DatasetIndex:
    """Lightweight scan of a dataset folder. Holds only well IDs and file paths."""

    root: Path
    wells: list[dict] = field(default_factory=list)  # [{well_id, split, hw, tw}]

    @classmethod
    def scan(cls, root: Path) -> "DatasetIndex":
        root = Path(root)
        idx = cls(root=root)
        for split in ("train", "test"):
            folder = root / split
            if not folder.is_dir():
                continue
            hw_paths = sorted(
                p for p in folder.glob("*__horizontal_well.csv") if "Zone" not in p.name
            )
            for hw in hw_paths:
                wid = hw.name.split("__", 1)[0]
                tw = folder / f"{wid}__typewell.csv"
                if not tw.exists():
                    continue
                idx.wells.append({"well_id": wid, "split": split, "hw": hw, "tw": tw})
        idx.wells.sort(key=lambda d: (d["split"], d["well_id"]))
        return idx

    def well_ids(self) -> list[str]:
        return [w["well_id"] for w in self.wells]

    @staticmethod
    def make_key(split: str, well_id: str) -> str:
        return f"{split}/{well_id}"

    def find(self, well_id: str, split: Optional[str] = None) -> Optional[dict]:
        for w in self.wells:
            if w["well_id"] != well_id:
                continue
            if split is not None and w["split"] != split:
                continue
            return w
        return None

    def find_key(self, key: str) -> Optional[dict]:
        if "/" not in key:
            return self.find(key)
        split, well_id = key.split("/", 1)
        return self.find(well_id, split=split)


def load_well(entry: dict) -> WellBundle:
    """Load one well bundle from a DatasetIndex entry."""
    hw = pd.read_csv(entry["hw"])
    tw = pd.read_csv(entry["tw"])
    return WellBundle(
        well_id=entry["well_id"],
        split=entry["split"],
        hw=hw,
        tw=tw,
        hw_path=entry["hw"],
        tw_path=entry["tw"],
    )


# ---------------------------------------------------------------------------
# Predictions CSV
# ---------------------------------------------------------------------------
@dataclass
class Predictions:
    """Parsed Kaggle-format predictions CSV (`id,tvt`)."""

    path: Path
    raw: pd.DataFrame  # columns: id, tvt, well_id, row_index
    by_well: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Predictions":
        path = Path(path)
        df = pd.read_csv(path)
        # tolerate case differences in column names
        cols_lower = {c.lower(): c for c in df.columns}
        if "id" not in cols_lower or "tvt" not in cols_lower:
            raise ValueError(
                f"Predictions CSV must have columns 'id' and 'tvt'. Got: {list(df.columns)}"
            )
        df = df.rename(columns={cols_lower["id"]: "id", cols_lower["tvt"]: "tvt"})
        df["well_id"] = df["id"].str.rsplit("_", n=1).str[0]
        df["row_index"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)
        by_well: dict[str, pd.DataFrame] = {}
        for wid, sub in df.groupby("well_id", sort=False):
            by_well[wid] = sub[["row_index", "tvt"]].reset_index(drop=True)
        return cls(path=path, raw=df, by_well=by_well)

    def for_well(self, well_id: str) -> Optional[pd.DataFrame]:
        return self.by_well.get(well_id)

    def coverage(self) -> str:
        n_wells = len(self.by_well)
        n_rows = len(self.raw)
        return f"{n_rows:,} rows across {n_wells} wells"


def predictions_aligned_to_hw(
    pred: pd.DataFrame, hw: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align a per-well predictions DataFrame to the horizontal-well DataFrame.

    Returns:
        md_array       (n,) MD values for the predicted rows
        pred_tvt_array (n,) predicted TVT values
        truth_array    (n,) corresponding truth TVT (NaN if no truth column)
    """
    md = hw["MD"].to_numpy()
    n = len(md)
    idx = pred["row_index"].to_numpy()
    valid = (idx >= 0) & (idx < n)
    idx = idx[valid]
    tvt_pred = pred["tvt"].to_numpy()[valid]
    md_arr = md[idx]
    if "TVT" in hw.columns:
        truth = hw["TVT"].to_numpy()[idx]
    else:
        truth = np.full(len(idx), np.nan)
    return md_arr, tvt_pred, truth


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE ignoring NaN pairs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff * diff)))
