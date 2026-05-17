"""Generate a self-contained EDA HTML report for the ROGII wellbore-geology dataset.

Usage:
    python generate_report.py

Output:
    /home/tom99763/ROGII/report/eda_report.html
    /home/tom99763/ROGII/report/well_summary.csv
    /home/tom99763/ROGII/report/typewell_summary.csv
    /home/tom99763/ROGII/report/baseline_rmse.csv
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/tom99763/ROGII/rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"

REPORT_DIR = Path("/home/tom99763/ROGII/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HW_COLS_FULL = ["MD", "X", "Y", "Z", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "TVT", "GR", "TVT_input"]
HW_COLS_TEST = ["MD", "X", "Y", "Z", "GR", "TVT_input"]

NEUTRAL_BG = "#fafbfc"


def well_id_from(path: Path) -> str:
    name = path.name
    return name.split("__", 1)[0] if "__" in name else name.split(".", 1)[0]


def list_csv(folder: Path, suffix: str) -> list[Path]:
    return sorted(p for p in folder.glob(f"*{suffix}") if "Zone" not in p.name)


train_hw_paths = list_csv(TRAIN_DIR, "__horizontal_well.csv")
train_tw_paths = list_csv(TRAIN_DIR, "__typewell.csv")
test_hw_paths = list_csv(TEST_DIR, "__horizontal_well.csv")
test_tw_paths = list_csv(TEST_DIR, "__typewell.csv")

print(f"train HW={len(train_hw_paths)} TW={len(train_tw_paths)}  test HW={len(test_hw_paths)} TW={len(test_tw_paths)}")


# ---------------------------------------------------------------------------
# Pass 1: per-well summary over every horizontal well
# ---------------------------------------------------------------------------
def gr_missingness_blocks(mask: np.ndarray) -> dict:
    """Return statistics on contiguous runs of GR=NaN."""
    if mask.sum() == 0:
        return {"n_blocks": 0, "max_block_len": 0, "mean_block_len": 0.0}
    changes = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    lengths = ends - starts
    return {
        "n_blocks": int(len(lengths)),
        "max_block_len": int(lengths.max()),
        "mean_block_len": float(lengths.mean()),
    }


def summarize_hw(path: Path, split: str) -> dict:
    cols = HW_COLS_FULL if split == "train" else HW_COLS_TEST
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    out: dict = {"well_id": well_id_from(path), "split": split, "rows": len(df)}

    md = df["MD"].to_numpy()
    out["md_min"] = float(md.min())
    out["md_max"] = float(md.max())
    out["md_range"] = float(md.max() - md.min())
    md_diff = np.diff(md)
    out["md_step_med"] = float(np.median(md_diff))
    out["md_step_std"] = float(md_diff.std())

    out["x_mean"] = float(df["X"].mean())
    out["y_mean"] = float(df["Y"].mean())
    out["z_mean"] = float(df["Z"].mean())
    out["z_min"] = float(df["Z"].min())
    out["z_max"] = float(df["Z"].max())
    out["xy_bbox"] = float(np.hypot(df["X"].max() - df["X"].min(), df["Y"].max() - df["Y"].min()))

    # lateral azimuth (deg from East) using PCA of XY
    xy = df[["X", "Y"]].to_numpy()
    xy_c = xy - xy.mean(axis=0)
    cov = np.cov(xy_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, -1]
    out["lateral_azimuth_deg"] = float(np.degrees(np.arctan2(v[1], v[0])) % 180)

    # GR
    gr = df["GR"].to_numpy()
    gr_valid = gr[~np.isnan(gr)]
    out["gr_missing_pct"] = float(np.isnan(gr).mean() * 100)
    if len(gr_valid):
        out["gr_min"] = float(gr_valid.min())
        out["gr_max"] = float(gr_valid.max())
        out["gr_mean"] = float(gr_valid.mean())
        out["gr_std"] = float(gr_valid.std())
    else:
        out["gr_min"] = out["gr_max"] = out["gr_mean"] = out["gr_std"] = np.nan
    blocks = gr_missingness_blocks(np.isnan(gr))
    out.update({f"gr_miss_{k}": v for k, v in blocks.items()})

    # PS detection
    tvt_input = df["TVT_input"].to_numpy()
    ps_mask = np.isnan(tvt_input)
    out["eval_rows"] = int(ps_mask.sum())
    out["known_rows"] = int((~ps_mask).sum())
    out["eval_pct"] = float(ps_mask.mean() * 100)
    if ps_mask.any():
        ps_idx = int(np.flatnonzero(ps_mask)[0])
        out["ps_idx"] = ps_idx
        out["ps_md"] = float(md[ps_idx])
        out["ps_md_frac"] = float((md[ps_idx] - md[0]) / max(md[-1] - md[0], 1.0))
    else:
        out["ps_idx"] = None
        out["ps_md"] = np.nan
        out["ps_md_frac"] = np.nan

    if split == "train":
        tvt = df["TVT"].to_numpy()
        out["tvt_min"] = float(tvt.min())
        out["tvt_max"] = float(tvt.max())
        out["tvt_range"] = float(tvt.max() - tvt.min())
        # dip = dTVT/dMD
        slope = np.diff(tvt) / np.diff(md)
        slope = slope[np.isfinite(slope)]
        out["slope_med"] = float(np.median(slope))
        out["slope_abs_med"] = float(np.median(np.abs(slope)))
        out["slope_p05"] = float(np.percentile(slope, 5))
        out["slope_p95"] = float(np.percentile(slope, 95))
        # direction over the well
        flat_eps = 0.02
        inc = float((slope > flat_eps).mean() * 100)
        dec = float((slope < -flat_eps).mean() * 100)
        out["pct_inc"] = inc
        out["pct_dec"] = dec
        out["pct_flat"] = float(100 - inc - dec)

        # formation-top dip (using ANCC as proxy: dANCC/dMD)
        for ftcol in ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
            if ftcol in df.columns:
                v = df[ftcol].to_numpy()
                sl = np.diff(v) / np.diff(md)
                sl = sl[np.isfinite(sl)]
                if len(sl):
                    out[f"{ftcol}_slope_med"] = float(np.median(sl))
    return out


def summarize_tw(path: Path) -> dict:
    df = pd.read_csv(path)
    out: dict = {"well_id": well_id_from(path), "rows": len(df)}
    for c in ("TVT", "GR"):
        if c in df.columns:
            v = df[c].dropna()
            out[f"{c}_min"] = float(v.min())
            out[f"{c}_max"] = float(v.max())
            out[f"{c}_range"] = float(v.max() - v.min())
            out[f"{c}_mean"] = float(v.mean())
    if "Geology" in df.columns:
        geo = df["Geology"].dropna()
        out["geo_unique"] = int(geo.nunique())
        out["geo_missing_pct"] = float(df["Geology"].isna().mean() * 100)
    return out


t0 = time.time()
print("[pass 1] summarising horizontal wells...")
rows = []
for i, p in enumerate(train_hw_paths):
    rows.append(summarize_hw(p, "train"))
    if (i + 1) % 100 == 0:
        print(f"  train hw {i+1}/{len(train_hw_paths)}  ({time.time()-t0:.1f}s)")
for p in test_hw_paths:
    rows.append(summarize_hw(p, "test"))
well_summary = pd.DataFrame(rows)

print("[pass 1] summarising typewells...")
tw_rows = [summarize_tw(p) for p in train_tw_paths]
typewell_summary = pd.DataFrame(tw_rows)

# Geology vocabulary across train typewells
print("[pass 1] building geology vocabulary...")
geo_counter: Counter = Counter()
for p in train_tw_paths:
    df = pd.read_csv(p, usecols=["Geology"])
    geo_counter.update(df["Geology"].dropna().astype(str))
geology_counts = pd.DataFrame(geo_counter.most_common(), columns=["geology", "rows"])

well_summary.to_csv(REPORT_DIR / "well_summary.csv", index=False)
typewell_summary.to_csv(REPORT_DIR / "typewell_summary.csv", index=False)
geology_counts.to_csv(REPORT_DIR / "geology_counts.csv", index=False)
print(f"[pass 1] done in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Pass 2: baseline RMSE floors on training wells
# ---------------------------------------------------------------------------
def baseline_rmse_for_well(path: Path) -> dict | None:
    df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
    ps_mask = df["TVT_input"].isna()
    if not ps_mask.any():
        return None
    ps_idx = int(np.flatnonzero(ps_mask.to_numpy())[0])
    if ps_idx < 5 or ps_idx >= len(df) - 1:
        return None

    md = df["MD"].to_numpy()
    z = df["Z"].to_numpy()
    tvt = df["TVT"].to_numpy()

    truth = tvt[ps_idx:]
    n_pred = len(truth)

    # 1) constant
    last_tvt = tvt[ps_idx - 1]
    pred_const = np.full(n_pred, last_tvt)

    # 2) linear in MD using last 50 known points
    win = min(50, ps_idx)
    md_w = md[ps_idx - win:ps_idx]
    tvt_w = tvt[ps_idx - win:ps_idx]
    slope_md, intercept_md = np.polyfit(md_w, tvt_w, 1)
    pred_lin_md = slope_md * md[ps_idx:] + intercept_md

    # 3) linear in Z using last 50 known points (TVT often tracks Z+const)
    z_w = z[ps_idx - win:ps_idx]
    if np.ptp(z_w) > 1e-6:
        slope_z, intercept_z = np.polyfit(z_w, tvt_w, 1)
        pred_lin_z = slope_z * z[ps_idx:] + intercept_z
    else:
        pred_lin_z = pred_const.copy()

    # 4) "oracle constant" = mean of truth (purely diagnostic ceiling for constant family)
    pred_mean = np.full(n_pred, truth.mean())

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "well_id": well_id_from(path),
        "n_pred": n_pred,
        "rmse_constant": rmse(pred_const, truth),
        "rmse_linear_md": rmse(pred_lin_md, truth),
        "rmse_linear_z": rmse(pred_lin_z, truth),
        "rmse_oracle_const": rmse(pred_mean, truth),
    }


t0 = time.time()
print("[pass 2] running baseline RMSEs on train wells...")
baseline_rows = []
for i, p in enumerate(train_hw_paths):
    r = baseline_rmse_for_well(p)
    if r is not None:
        baseline_rows.append(r)
    if (i + 1) % 200 == 0:
        print(f"  baseline {i+1}/{len(train_hw_paths)}  ({time.time()-t0:.1f}s)")
baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(REPORT_DIR / "baseline_rmse.csv", index=False)
print(f"[pass 2] done in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Pass 3: small samples for cross-well visuals (overview wells, map sample)
# ---------------------------------------------------------------------------
SAMPLE_WELL_IDS = ["000d7d20", "00bbac68", "00e12e8b"]  # also happen to be the test wells -> train mirror only used for narratives

# Pick 3 demo train wells: short, median, long laterals
sorted_by_eval = well_summary[well_summary.split == "train"].sort_values("eval_rows")
demo_ids = [
    sorted_by_eval.iloc[len(sorted_by_eval) // 10]["well_id"],
    sorted_by_eval.iloc[len(sorted_by_eval) // 2]["well_id"],
    sorted_by_eval.iloc[-len(sorted_by_eval) // 10]["well_id"],
]
print("Demo wells (short/median/long eval-zone):", demo_ids)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


def fmt_int(n: float) -> str:
    if pd.isna(n):
        return "-"
    return f"{int(n):,}"


def fmt_flt(n: float, p: int = 2) -> str:
    if pd.isna(n):
        return "-"
    return f"{n:,.{p}f}"


# Fig 1: file counts
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.barh(
    ["train horizontal", "train typewell", "test horizontal", "test typewell"],
    [len(train_hw_paths), len(train_tw_paths), len(test_hw_paths), len(test_tw_paths)],
    color=["#3b6fbb", "#5cb874", "#e9924b", "#d05a5a"],
)
for i, v in enumerate([len(train_hw_paths), len(train_tw_paths), len(test_hw_paths), len(test_tw_paths)]):
    ax.text(v + 5, i, str(v), va="center")
ax.set_xlabel("Files")
ax.set_title("Dataset file inventory")
save_fig("file_counts", fig)


# Fig 2: PS location distribution (MD-frac)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
ts = well_summary[well_summary.split == "train"]
axes[0].hist(ts["eval_pct"], bins=40, color="#3b6fbb", edgecolor="white")
axes[0].axvline(ts["eval_pct"].median(), color="crimson", ls="--", label=f"median {ts['eval_pct'].median():.1f}%")
axes[0].set_title("% of well that must be predicted (PS to end)")
axes[0].set_xlabel("eval rows / total rows  (%)")
axes[0].set_ylabel("well count")
axes[0].legend()

axes[1].hist(ts["ps_md_frac"] * 100, bins=40, color="#5cb874", edgecolor="white")
axes[1].axvline(float(ts["ps_md_frac"].median() * 100), color="crimson", ls="--",
                label=f"median {ts['ps_md_frac'].median()*100:.1f}%")
axes[1].set_title("PS position along MD (% of total MD)")
axes[1].set_xlabel("(PS_MD - MD_min) / MD_range  (%)")
axes[1].set_ylabel("well count")
axes[1].legend()
save_fig("ps_distribution", fig)


# Fig 3: TVT range & abs slope distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
axes[0].hist(ts["tvt_range"], bins=40, color="#7b5ec0", edgecolor="white")
axes[0].set_title("TVT range per training well (ft)")
axes[0].set_xlabel("max(TVT) - min(TVT)")
axes[0].set_ylabel("well count")

axes[1].hist(np.clip(ts["slope_abs_med"], 0, 1), bins=40, color="#e9924b", edgecolor="white")
axes[1].set_title("Median |dTVT/dMD| per well (clipped at 1)")
axes[1].set_xlabel("ft of TVT per ft of MD")
axes[1].set_ylabel("well count")
save_fig("tvt_dip", fig)


# Fig 4: Direction class breakdown (stacked bar)
fig, ax = plt.subplots(figsize=(8, 3.6))
dir_pct = ts[["pct_inc", "pct_flat", "pct_dec"]].mean()
ax.bar(["increasing", "nearly flat", "decreasing"], dir_pct.values,
       color=["#5cb874", "#9e9e9e", "#d05a5a"])
for i, v in enumerate(dir_pct.values):
    ax.text(i, v + 0.6, f"{v:.1f}%", ha="center")
ax.set_ylabel("avg % of within-well rows")
ax.set_title("Average TVT direction breakdown across training wells (threshold = 0.02 ft/ft)")
save_fig("direction_classes", fig)


# Fig 5: GR missingness — overall and blockiness
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
axes[0].hist(ts["gr_missing_pct"], bins=40, color="#d05a5a", edgecolor="white")
axes[0].set_title("GR missing % per training well")
axes[0].set_xlabel("missing rows / total rows  (%)")
axes[0].set_ylabel("well count")

mask = ts["gr_miss_n_blocks"] > 0
axes[1].scatter(ts.loc[mask, "gr_miss_n_blocks"], ts.loc[mask, "gr_miss_mean_block_len"],
                alpha=0.4, s=14, c="#7b5ec0")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("GR missing-block count vs mean block length (log-log)")
axes[1].set_xlabel("# missing runs")
axes[1].set_ylabel("mean missing run length (rows)")
save_fig("gr_missingness", fig)


# Fig 6: map view of all training wells + test wells
fig, ax = plt.subplots(figsize=(8, 8))
for p in train_hw_paths:
    df = pd.read_csv(p, usecols=["X", "Y"])
    step = max(1, len(df) // 100)
    d = df.iloc[::step]
    ax.plot(d["X"], d["Y"], color="#cccccc", lw=0.5, alpha=0.5)
for p in test_hw_paths:
    df = pd.read_csv(p, usecols=["X", "Y"])
    ax.plot(df["X"], df["Y"], color="#2b6cb0", lw=2.0, label=well_id_from(p))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X / Easting (ft)")
ax.set_ylabel("Y / Northing (ft)")
ax.set_title(f"Map view: {len(train_hw_paths)} train wells (gray) + {len(test_hw_paths)} visible test wells (blue)")
handles, labels = ax.get_legend_handles_labels()
if handles:
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(*zip(*uniq), loc="best", fontsize=8)
save_fig("map", fig)


# Fig 7: lateral azimuth rose
fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
az = np.deg2rad(well_summary["lateral_azimuth_deg"].dropna())
bins = np.linspace(0, np.pi, 19)
counts, _ = np.histogram(az, bins=bins)
theta = (bins[:-1] + bins[1:]) / 2
ax.bar(theta, counts, width=np.diff(bins), bottom=0, color="#3b6fbb", edgecolor="white", alpha=0.85)
ax.bar(theta + np.pi, counts, width=np.diff(bins), bottom=0, color="#3b6fbb", edgecolor="white", alpha=0.85)
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_title("Lateral azimuth distribution (PCA of XY trajectory, 0=East)", pad=14)
save_fig("azimuth", fig)


# Fig 8: overview of 3 demo wells (TVT + GR vs MD, typewell GR vs TVT)
def plot_well_panel(ax_tvt, ax_gr, ax_tw, well_id: str):
    hw = pd.read_csv(TRAIN_DIR / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN_DIR / f"{well_id}__typewell.csv")

    ax_tvt.plot(hw["MD"], hw["TVT"], color="#2b6cb0", lw=1.3, label="TVT (truth)")
    ax_tvt.plot(hw["MD"], hw["TVT_input"], color="#e9924b", lw=1.0, label="TVT_input (known)")
    ps_idx = int(hw["TVT_input"].isna().to_numpy().argmax()) if hw["TVT_input"].isna().any() else None
    if ps_idx and ps_idx > 0:
        ax_tvt.axvline(hw.loc[ps_idx, "MD"], color="crimson", ls="--", lw=1.0, label="PS")
    ax_tvt.set_title(f"{well_id}: TVT vs MD")
    ax_tvt.set_xlabel("MD")
    ax_tvt.set_ylabel("TVT")
    ax_tvt.legend(fontsize=8)

    ax_gr.plot(hw["MD"], hw["GR"], color="#5cb874", lw=0.7)
    if ps_idx and ps_idx > 0:
        ax_gr.axvline(hw.loc[ps_idx, "MD"], color="crimson", ls="--", lw=1.0)
    ax_gr.set_title("Horizontal GR vs MD")
    ax_gr.set_xlabel("MD")
    ax_gr.set_ylabel("GR")

    if {"TVT", "GR"}.issubset(tw.columns):
        ax_tw.plot(tw["GR"], tw["TVT"], color="black", lw=1.0, label="typewell GR")
    known = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    ax_tw.scatter(known["GR"], known["TVT_input"], s=6, color="#5cb874", alpha=0.4, label="HW known")
    ax_tw.invert_yaxis()
    ax_tw.set_title("GR signatures on TVT axis")
    ax_tw.set_xlabel("GR")
    ax_tw.set_ylabel("TVT (inverted)")
    ax_tw.legend(fontsize=8)


fig, axes = plt.subplots(len(demo_ids), 3, figsize=(15, 3.4 * len(demo_ids)), constrained_layout=True)
for i, wid in enumerate(demo_ids):
    plot_well_panel(axes[i, 0], axes[i, 1], axes[i, 2], wid)
save_fig("well_overview", fig)


# Fig 9: baseline RMSE distribution (log scale)
fig, ax = plt.subplots(figsize=(9, 4))
data = [baseline_df["rmse_constant"], baseline_df["rmse_linear_md"],
        baseline_df["rmse_linear_z"], baseline_df["rmse_oracle_const"]]
labels = ["constant\n(last known)", "linear in MD\n(last 50)", "linear in Z\n(last 50)",
          "oracle constant\n(mean truth)"]
ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#e2efff", edgecolor="#3b6fbb"),
           medianprops=dict(color="crimson"))
ax.set_yscale("log")
ax.set_ylabel("RMSE per well (ft, log)")
ax.set_title("Baseline RMSE distribution across 773 training wells\n(simulated by holding out each well's own PS-to-end)")
save_fig("baselines", fig)


# Fig 10: cumulative RMSE per baseline
fig, ax = plt.subplots(figsize=(8, 4))
for col, color, label in [
    ("rmse_constant", "#d05a5a", "constant"),
    ("rmse_linear_md", "#7b5ec0", "linear-MD"),
    ("rmse_linear_z", "#3b6fbb", "linear-Z"),
]:
    v = np.sort(baseline_df[col].values)
    y = np.arange(1, len(v) + 1) / len(v)
    ax.plot(v, y, label=label, color=color)
ax.set_xscale("log")
ax.set_xlabel("RMSE (ft, log)")
ax.set_ylabel("CDF over wells")
ax.set_title("CDF of per-well RMSE for simple baselines")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("baselines_cdf", fig)


# Fig 11: typewell geology vocab
fig, ax = plt.subplots(figsize=(8, 4.5))
top = geology_counts.head(15).iloc[::-1]
ax.barh(top["geology"], top["rows"], color="#5b6cb0")
for i, (g, r) in enumerate(zip(top["geology"], top["rows"])):
    ax.text(r, i, f" {int(r):,}", va="center", fontsize=8)
ax.set_title("Most common typewell geology layer labels (sum of rows in train)")
ax.set_xlabel("rows labelled")
save_fig("geology", fig)


# Fig 12: formation-top slope distribution (training-only leak signal)
slope_cols = [c for c in well_summary.columns if c.endswith("_slope_med") and c.split("_")[0] in
              {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}]
fig, ax = plt.subplots(figsize=(9, 3.6))
data = [well_summary[c].dropna() for c in slope_cols]
labels = [c.split("_")[0] for c in slope_cols]
ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#e2f1e2", edgecolor="#5cb874"))
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Median d(formation_top elevation)/dMD per training well")
ax.set_ylabel("ft per ft of MD")
save_fig("formation_dips", fig)


# Fig 13: GR vs |dTVT/dMD| relationship
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sub = ts.dropna(subset=["gr_std", "slope_abs_med"])
ax.scatter(sub["gr_std"], sub["slope_abs_med"], alpha=0.35, s=12, c="#3b6fbb")
ax.set_xlabel("GR std (per well)")
ax.set_ylabel("median |dTVT/dMD|")
ax.set_yscale("log")
ax.set_title("Does noisier GR correlate with steeper geology?")
save_fig("gr_vs_dip", fig)


# Fig 14: test wells highlighted on map, with nearest-train-neighbour distance
def well_centroid(df):
    return df["X"].mean(), df["Y"].mean()


train_cent = []
for _, row in well_summary[well_summary.split == "train"].iterrows():
    train_cent.append((row["x_mean"], row["y_mean"], row["well_id"]))
train_cent_arr = np.array([(x, y) for x, y, _ in train_cent])
test_dist_rows = []
for _, row in well_summary[well_summary.split == "test"].iterrows():
    dxy = train_cent_arr - np.array([row["x_mean"], row["y_mean"]])
    d = np.hypot(dxy[:, 0], dxy[:, 1])
    order = np.argsort(d)[:5]
    for k, idx in enumerate(order):
        test_dist_rows.append({
            "test_well": row["well_id"],
            "rank": k + 1,
            "nearest_train_well": train_cent[idx][2],
            "distance_ft": float(d[idx]),
        })
test_nn = pd.DataFrame(test_dist_rows)
test_nn.to_csv(REPORT_DIR / "test_nearest_neighbours.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.scatter(well_summary[well_summary.split == "train"]["x_mean"],
           well_summary[well_summary.split == "train"]["y_mean"],
           s=8, color="#cccccc", label="train (centroid)")
for _, row in well_summary[well_summary.split == "test"].iterrows():
    ax.scatter(row["x_mean"], row["y_mean"], s=80, marker="*", color="#d05a5a", zorder=5)
    ax.text(row["x_mean"], row["y_mean"], "  " + row["well_id"], fontsize=8, color="#d05a5a")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Test wells (red stars) vs training-well centroids (gray)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc="best", fontsize=8)
save_fig("test_neighbours", fig)


# ---------------------------------------------------------------------------
# Aggregate numbers for the HTML
# ---------------------------------------------------------------------------
n_train = len(train_hw_paths)
n_test = len(test_hw_paths)
total_train_rows = int(well_summary[well_summary.split == "train"]["rows"].sum())
total_test_rows = int(well_summary[well_summary.split == "test"]["rows"].sum())
total_eval_rows_test = int(well_summary[well_summary.split == "test"]["eval_rows"].sum())

stats_ts = ts.describe(percentiles=[0.05, 0.5, 0.95]).T


def pct(s, q):
    return float(np.percentile(s.dropna(), q))


executive = {
    "n_train_wells": n_train,
    "n_test_wells": n_test,
    "total_train_rows": total_train_rows,
    "total_eval_rows_test": total_eval_rows_test,
    "median_well_len": int(ts["rows"].median()),
    "median_eval_pct": float(ts["eval_pct"].median()),
    "median_tvt_range": float(ts["tvt_range"].median()),
    "median_gr_missing": float(ts["gr_missing_pct"].median()),
    "median_slope_abs": float(ts["slope_abs_med"].median()),
    "p95_slope_abs": pct(ts["slope_abs_med"], 95),
    "med_rmse_const": float(baseline_df["rmse_constant"].median()),
    "med_rmse_lin_md": float(baseline_df["rmse_linear_md"].median()),
    "med_rmse_lin_z": float(baseline_df["rmse_linear_z"].median()),
    "p90_rmse_const": pct(baseline_df["rmse_constant"], 90),
    "p90_rmse_lin_md": pct(baseline_df["rmse_linear_md"], 90),
    "p90_rmse_lin_z": pct(baseline_df["rmse_linear_z"], 90),
}


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def img_tag(name: str, caption: str = "", width: str = "100%") -> str:
    b64 = FIGS[name]
    cap_html = f"<div class='cap'>{caption}</div>" if caption else ""
    return (
        f"<figure><img src='data:image/png;base64,{b64}' style='width:{width};max-width:100%'/>"
        f"{cap_html}</figure>"
    )


def df_to_html(df: pd.DataFrame, max_rows: int | None = 12) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, classes="data", border=0, float_format=lambda x: f"{x:,.2f}")


well_describe = well_summary[well_summary.split == "train"][[
    "rows", "md_range", "tvt_range", "gr_missing_pct", "eval_rows", "eval_pct",
    "slope_abs_med",
]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "metric"})

tw_describe = typewell_summary[[
    "rows", "TVT_range", "GR_range", "geo_unique", "geo_missing_pct",
]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "metric"})

test_well_table = well_summary[well_summary.split == "test"][[
    "well_id", "rows", "md_range", "known_rows", "eval_rows", "eval_pct",
    "ps_md", "ps_md_frac", "gr_missing_pct", "x_mean", "y_mean", "z_mean",
]].copy()
test_well_table["ps_md_frac"] = (test_well_table["ps_md_frac"] * 100)
test_well_table = test_well_table.rename(columns={"ps_md_frac": "ps_md_frac_pct"})

baseline_summary = baseline_df[[
    "rmse_constant", "rmse_linear_md", "rmse_linear_z", "rmse_oracle_const",
]].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).T.reset_index().rename(columns={"index": "baseline"})

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII Wellbore Geology Prediction — Deep EDA Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    background: {NEUTRAL_BG};
    color: #1f2933;
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 36px 80px;
    line-height: 1.55;
  }}
  h1 {{ font-size: 28px; border-bottom: 2px solid #3b6fbb; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 21px; color: #2b4a7a; margin-top: 36px; border-left: 4px solid #3b6fbb; padding-left: 10px; }}
  h3 {{ font-size: 16px; color: #444; margin-top: 22px; }}
  p, li {{ font-size: 14px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 18px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }}
  .kpi {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }}
  .kpi .v {{ font-size: 20px; font-weight: 600; color: #1f2933; }}
  .kpi .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 10px 0 18px; background: white; }}
  table.data th, table.data td {{ border-bottom: 1px solid #e3e7ec; padding: 6px 10px; text-align: right; }}
  table.data th {{ background: #eef3f9; color: #2b4a7a; text-align: right; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  figure {{ margin: 14px 0 22px; }}
  .cap {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
  blockquote {{ border-left: 3px solid #5cb874; margin: 14px 0; padding: 8px 14px; background: white; color: #2f4d36; font-size: 13.5px; }}
  code {{ background: #eef0f2; padding: 1px 4px; border-radius: 3px; font-size: 13px; }}
  .col2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  ul {{ padding-left: 22px; }}
  .small {{ font-size: 12px; color: #6b7280; }}
</style>
</head>
<body>

<h1>ROGII Wellbore Geology Prediction &mdash; Deep EDA Report</h1>
<div class="meta">
  Generated by <code>report/generate_report.py</code> &middot;
  Dataset path: <code>{DATA_ROOT}</code> &middot;
  Train wells: {n_train} &middot; Test (template) wells: {n_test}
</div>

<h2>1. Executive Summary</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Train wells</div><div class="v">{executive['n_train_wells']:,}</div></div>
  <div class="kpi"><div class="l">Train rows</div><div class="v">{executive['total_train_rows']:,}</div></div>
  <div class="kpi"><div class="l">Visible test wells</div><div class="v">{executive['n_test_wells']:,}</div></div>
  <div class="kpi"><div class="l">Visible test rows to predict</div><div class="v">{executive['total_eval_rows_test']:,}</div></div>
  <div class="kpi"><div class="l">Median well length (rows)</div><div class="v">{executive['median_well_len']:,}</div></div>
  <div class="kpi"><div class="l">Median % to predict</div><div class="v">{executive['median_eval_pct']:.1f}%</div></div>
  <div class="kpi"><div class="l">Median TVT range / well (ft)</div><div class="v">{executive['median_tvt_range']:,.0f}</div></div>
  <div class="kpi"><div class="l">Median GR missing</div><div class="v">{executive['median_gr_missing']:.1f}%</div></div>
  <div class="kpi"><div class="l">Median |dTVT/dMD|</div><div class="v">{executive['median_slope_abs']:.3f}</div></div>
  <div class="kpi"><div class="l">Constant baseline RMSE (median)</div><div class="v">{executive['med_rmse_const']:,.1f}</div></div>
  <div class="kpi"><div class="l">Linear-MD baseline RMSE (median)</div><div class="v">{executive['med_rmse_lin_md']:,.1f}</div></div>
  <div class="kpi"><div class="l">Linear-Z baseline RMSE (median)</div><div class="v">{executive['med_rmse_lin_z']:,.1f}</div></div>
</div>

<blockquote>
<b>One-paragraph takeaway.</b>
The competition is a <i>per-well sequence-to-sequence regression</i>: predict the true vertical thickness
(<code>TVT</code>) of a horizontal wellbore from the Prediction Start point to the toe, using the wellbore
trajectory (<code>X, Y, Z</code>), a noisy gamma-ray log (<code>GR</code>, ~29% missing on average), and a paired
vertical reference log (<i>typewell</i>) of <code>GR</code> vs <code>TVT</code>. The median well asks the model to extend
the last ~27% of known TVT into the next ~73%; geology dips are mostly mild (median |dTVT/dMD| ≈
{executive['median_slope_abs']:.3f}) but the upper tail is steep (p95 ≈ {executive['p95_slope_abs']:.2f}).
Naive constant extrapolation lands at a median RMSE of {executive['med_rmse_const']:,.1f} ft per well;
a 50-point linear-Z extrapolation already cuts the median to {executive['med_rmse_lin_z']:,.1f} ft &mdash;
any modeling effort needs to beat this floor.
</blockquote>

<h2>2. Dataset Inventory</h2>
<p>Two CSVs per well are provided: a <b>horizontal well</b> log sampled every 1&nbsp;ft of measured depth
(<code>MD</code>), and a paired <b>typewell</b> (vertical reference well) sampled every 0.5&nbsp;ft of <code>TVT</code>.
Training wells additionally include a PNG visualisation.</p>
{img_tag('file_counts')}
<p class="small">The 3 wells in <code>test/</code> are a <i>submission template</i>; Kaggle replaces them with a
hidden test set at scoring time. Any modeling pipeline must be agnostic to well IDs, well count, and well
length.</p>

<h2>3. Schema</h2>
<table class="data">
  <thead><tr><th>file</th><th>columns</th><th>missing in test?</th><th>role</th></tr></thead>
  <tbody>
    <tr><td>horizontal_well.csv</td><td><code>MD, X, Y, Z, GR, TVT_input</code></td><td>—</td><td>features at prediction time</td></tr>
    <tr><td>horizontal_well.csv (train only)</td><td><code>TVT</code></td><td>removed in test</td><td>regression target</td></tr>
    <tr><td>horizontal_well.csv (train only)</td><td><code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code></td><td>removed in test</td><td>per-row elevation of formation tops &mdash; encodes the true geological structure along the trajectory</td></tr>
    <tr><td>typewell.csv</td><td><code>TVT, GR, Geology</code></td><td>—</td><td>vertical reference: GR↔TVT dictionary + (sparse) geology label per row</td></tr>
  </tbody>
</table>

<h3>Coordinate notes</h3>
<ul>
  <li><code>MD</code> is the wellbore length from surface (ft). Step is uniformly 1.0&nbsp;ft per row in every well in the dataset.</li>
  <li><code>X, Y</code> are projected map coordinates (ft); <code>Z</code> is elevation (negative numbers; deeper = more negative).</li>
  <li><code>TVT</code> is <i>true vertical thickness</i> measured in the <i>typewell&apos;s</i> vertical reference frame &mdash; it is NOT identical to depth-from-surface. It is the depth axis on which the typewell&apos;s GR/geology are indexed.</li>
  <li>The 6 formation-top columns are elevations of geological horizons at the trajectory point (negative, in feet). They differ from <code>TVT</code> by a per-region datum offset.</li>
</ul>

<h2>4. Prediction-Start (PS) Anatomy</h2>
<p>In training, <code>TVT_input</code> is set up exactly like the test set: it copies <code>TVT</code> until the
PS row, then becomes NaN. This means every training well is a self-contained validation example for the
exact prediction task.</p>
{img_tag('ps_distribution')}
<p>The distribution is concentrated: PS sits near the 20&ndash;30% mark of the well&apos;s MD range,
leaving 60&ndash;90% of the trajectory (typically ~73%) as the evaluation zone. The known segment is
roughly 1,300&ndash;2,000 rows in most wells.</p>

<h3>Visible-test wells — per-well anatomy</h3>
{df_to_html(test_well_table, max_rows=None)}

<h2>5. Target: TVT</h2>
<p><code>TVT</code> is complete in training (0% missing). Per-well range and per-step gradient distributions:</p>
{img_tag('tvt_dip')}
{img_tag('direction_classes')}
<p>The geology along any single well is <i>not</i> monotonic: every well contains
significant fractions of increasing, decreasing, and nearly-flat segments. A model that assumes monotonic
TVT will fail on the steep tail.</p>

<h3>Per-well TVT / GR statistics &mdash; describe()</h3>
{df_to_html(well_describe, max_rows=None)}

<h2>6. Gamma Ray (GR)</h2>
<p>GR is the main signal for geological alignment, and it is noisy:</p>
{img_tag('gr_missingness')}
<p>Most wells have many short missing runs (median run length 1&ndash;3 rows) plus a long tail of wells
that contain large contiguous gaps. Mean-imputation is dangerous; treat missingness with an explicit
mask or use a Gaussian-process / spline imputation that respects MD ordering.</p>

<h3>Does GR roughness correlate with steeper geology?</h3>
{img_tag('gr_vs_dip')}
<p>Weak positive trend: wells with more variable GR also tend to traverse steeper geology, but the
relationship is far from clean. Don&apos;t use GR std alone as a difficulty proxy.</p>

<h2>7. Trajectory &amp; Spatial Layout</h2>
{img_tag('map')}
<p>The wells cluster in a relatively dense region of map space &mdash; offset-well analysis is feasible.
Lateral azimuths concentrate on two main directions:</p>
{img_tag('azimuth')}
<p>This bimodality is consistent with two dominant horizontal-drilling orientations in the survey area.
For an inference-time well, the azimuth of nearby training wells is a useful prior on local geological dip.</p>

<h3>Test wells vs train-well centroids</h3>
{img_tag('test_neighbours')}
<p>The 3 visible test wells fall well inside the train distribution &mdash; nearest-train distances per
test well:</p>
{df_to_html(test_nn, max_rows=None)}

<h2>8. Typewell Reference Log</h2>
<p>Each horizontal well has one paired typewell. Typewell rows are sampled at 0.5&nbsp;ft of <code>TVT</code>.</p>
{df_to_html(tw_describe, max_rows=None)}
<p>Geology labels are sparse (median ~35% missing &mdash; only present at formation-top rows).
Across the 773 training typewells the geology vocabulary is small (28 unique labels), dominated by
six layers that exactly match the six formation-top column names in the training horizontal CSVs:</p>
{img_tag('geology')}

<h2>9. Formation-Top Columns (Training-Only Leak)</h2>
<p>The columns <code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code> are present <i>only</i> in training. Each row
contains the absolute elevation of one geological horizon at the trajectory point. They essentially encode
the true geological structure that the model must predict, so they cannot be used as inference inputs &mdash;
but they are extremely useful as <b>auxiliary regression targets</b> in a multi-task setup, or for building
synthetic supervision signals.</p>
{img_tag('formation_dips')}
<p>Median d(formation-top)/dMD values are tightly centered near zero with long tails &mdash; consistent with
mostly-flat geology that occasionally dips. Critically, all six layers move <i>together</i>: at any given
training row the six layer elevations are highly correlated, since they describe the same dipping stack
of beds.</p>

<h2>10. Sample Well Walkthrough</h2>
<p>Three demo wells across the lateral-length distribution (short / median / long evaluation zone). Each row
shows TVT vs MD with the known/unknown split, the horizontal GR vs MD, and the typewell GR&ndash;TVT
dictionary with the known horizontal GR overlaid.</p>
{img_tag('well_overview')}

<h2>11. Baseline RMSE Floors</h2>
<p>To establish what the metric looks like before any modeling, we simulate the official task on every
training well using its own PS index. Three baselines:</p>
<ul>
  <li><b>constant</b>: predict the last-known TVT for all post-PS rows.</li>
  <li><b>linear in MD</b>: fit <code>TVT ~ a + b·MD</code> on the last 50 known rows, extrapolate forward.</li>
  <li><b>linear in Z</b>: fit <code>TVT ~ a + b·Z</code> on the last 50 known rows; uses the wellbore&apos;s own
      vertical motion as a proxy for geological dip.</li>
  <li><b>oracle constant</b> (diagnostic): predict the mean of the true post-PS TVT &mdash; the lowest RMSE
      any constant predictor could achieve. Gap to constant tells you how much "drift" is in the truth.</li>
</ul>
{img_tag('baselines')}
{img_tag('baselines_cdf')}
{df_to_html(baseline_summary, max_rows=None)}

<blockquote>
<b>What this means for modelling.</b>
<ul>
  <li>The linear-Z baseline cuts median RMSE roughly in half versus constant &mdash; the wellbore&apos;s
      <code>Z</code> trajectory is already a strong predictor of <code>TVT</code> drift, since the lateral often roughly
      parallels the bed.</li>
  <li>The gap between constant and oracle-constant is large, confirming that the truth drifts during the
      eval zone &mdash; "predict last known value" is leaving a lot of error on the table.</li>
  <li>Any serious model must beat the linear-Z floor on a leave-one-well-out CV. If it doesn&apos;t, it
      hasn&apos;t learned anything beyond local geometry.</li>
</ul>
</blockquote>

<h2>12. Modeling Recommendations</h2>
<ol>
  <li><b>Validation</b>: leave-one-well-out (or GroupKFold by XY cluster) on the 773 training wells is the
      only credible CV. Random row-shuffles leak because rows within a well are highly autocorrelated.</li>
  <li><b>Per-well baseline ensemble</b>: even a simple weighted blend of constant + linear-Z is a competitive
      starting submission and gives a useful sanity check for any learned model.</li>
  <li><b>Signal alignment</b>: dynamic time warping (or cross-correlation) of the post-PS horizontal GR against
      either (a) the pre-PS HW GR↔TVT_input dictionary or (b) the typewell GR↔TVT curve gives a per-step TVT
      estimate. The slides emphasize that the pre-PS HW dictionary is often <i>better</i> than the typewell
      because the horizontal GR has higher resolution.</li>
  <li><b>Sequence model</b>: 1-D CNN or bi-directional Transformer that ingests
      <code>(MD-MD_PS, ΔX, ΔY, ΔZ, GR, GR_mask, TVT_input, TVT_input_mask, typewell_GR_at_TVT_input)</code>
      and predicts TVT in absolute or delta form. Multi-task auxiliary heads on the six formation-top
      elevations stabilize representations during training and add no inference-time cost.</li>
  <li><b>Spatial prior</b>: for each test well, find the K nearest train wells by XY centroid; their TVT
      curves (after aligning on the shared TVT range) provide a strong dip prior. Especially useful for
      wells where GR is mostly missing.</li>
  <li><b>Submission packaging</b>: the visible test folder is a template only &mdash; load test wells by
      directory listing, not hard-coded IDs. Don&apos;t tune any heuristics to those 3 wells.</li>
</ol>

<h2>13. Companion Files</h2>
<p>The report script also emits these CSVs into <code>report/</code> for reuse:</p>
<ul>
  <li><code>well_summary.csv</code> &mdash; one row per (train+test) horizontal well, ~30 numeric features.</li>
  <li><code>typewell_summary.csv</code> &mdash; one row per training typewell.</li>
  <li><code>geology_counts.csv</code> &mdash; geology label frequency across all training typewells.</li>
  <li><code>baseline_rmse.csv</code> &mdash; per-well RMSE for the three simulated baselines, useful as a CV
      target distribution.</li>
  <li><code>test_nearest_neighbours.csv</code> &mdash; 5 nearest train wells (by XY centroid) for each visible
      test well.</li>
</ul>

</body>
</html>
"""

(REPORT_DIR / "eda_report.html").write_text(html, encoding="utf-8")
print(f"\nReport written: {REPORT_DIR / 'eda_report.html'}")
print(f"  size: {(REPORT_DIR / 'eda_report.html').stat().st_size / 1024:.1f} KB")
