"""Generate report/index.html — the consolidated English knowledge base for the
ROGII Wellbore Geology Prediction work.

Bundles everything we have learned into one self-contained HTML file:
  * Dataset KPIs (read from the cached EDA artifacts).
  * Six named domain insights from community discussions + reference repos.
  * Method portfolio with verdicts.
  * Reference implementations (DTW notebook, LWD_inversion repo, Ambrus MDN poster).
  * The viewer tooling we built.
  * A runnable roadmap.

Reads existing artifacts in report/; does not re-run the heavy passes.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPORT_DIR = Path("/home/tom99763/ROGII/report")
REFS_DIR = REPORT_DIR / "refs"
REPO_URL = "https://github.com/tom99763/rogii-viewer"

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Cached EDA artifacts → headline numbers
# ---------------------------------------------------------------------------
ws = pd.read_csv(REPORT_DIR / "well_summary.csv")
br = pd.read_csv(REPORT_DIR / "baseline_rmse.csv")
gc = pd.read_csv(REPORT_DIR / "geology_counts.csv")
nn = pd.read_csv(REPORT_DIR / "test_nearest_neighbours.csv")
ts = ws[ws.split == "train"]

n_train = int((ws.split == "train").sum())
n_test = int((ws.split == "test").sum())
total_train_rows = int(ts["rows"].sum())
total_eval_rows_test = int(ws[ws.split == "test"]["eval_rows"].sum())
median_eval_pct = float(ts["eval_pct"].median())
median_tvt_range = float(ts["tvt_range"].median())
median_gr_missing = float(ts["gr_missing_pct"].median())
median_slope = float(ts["slope_abs_med"].median())
med_rmse_const = float(br["rmse_constant"].median())
med_rmse_lin_z = float(br["rmse_linear_z"].median())
top_geology = gc.head(6)["geology"].tolist()


# ---------------------------------------------------------------------------
# Mega-figure: dataset snapshot (4 quadrants)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax = axes[0, 0]
ax.hist(ts["eval_pct"], bins=40, color="#3b6fbb", edgecolor="white")
ax.axvline(ts["eval_pct"].median(), color="crimson", ls="--",
           label=f"median {ts['eval_pct'].median():.1f}%")
ax.set_title("(a) % of well that needs prediction")
ax.set_xlabel("eval %")
ax.set_ylabel("well count")
ax.legend()

ax = axes[0, 1]
data = [br["rmse_constant"], br["rmse_linear_md"], br["rmse_linear_z"], br["rmse_oracle_const"]]
labels = ["constant", "linear-MD", "linear-Z", "oracle\nconst"]
ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#e2efff", edgecolor="#3b6fbb"),
           medianprops=dict(color="crimson"))
ax.set_yscale("log")
ax.set_ylabel("RMSE (ft, log scale)")
ax.set_title("(b) Baseline RMSE floors\n(LOWO-simulated on training wells)")

ax = axes[1, 0]
ax.hist(ts["gr_missing_pct"], bins=40, color="#d05a5a", edgecolor="white")
ax.axvline(ts["gr_missing_pct"].median(), color="black", ls="--",
           label=f"median {ts['gr_missing_pct'].median():.1f}%")
ax.set_title("(c) GR missing rate per well")
ax.set_xlabel("missing %")
ax.set_ylabel("well count")
ax.legend()

ax = axes[1, 1]
ax.hist(np.clip(ts["slope_abs_med"], 0, 1), bins=40, color="#7b5ec0", edgecolor="white")
ax.axvline(ts["slope_abs_med"].median(), color="black", ls="--",
           label=f"median {ts['slope_abs_med'].median():.3f}")
ax.set_title("(d) Median |dTVT/dMD| per well (dip steepness)")
ax.set_xlabel("ft of TVT per ft of MD")
ax.set_ylabel("well count")
ax.legend()
save_fig("snapshot", fig)


# Map view
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(ts["x_mean"], ts["y_mean"], s=6, c="#cccccc", label="train (centroid)")
te = ws[ws.split == "test"]
for _, row in te.iterrows():
    ax.scatter(row["x_mean"], row["y_mean"], s=80, marker="*", c="#d05a5a", zorder=5)
    ax.text(row["x_mean"], row["y_mean"], "  " + row["well_id"], fontsize=8, color="#d05a5a")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Visible test wells (red) within training well centroids (gray)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc="best", fontsize=8)
save_fig("map", fig)


# ---------------------------------------------------------------------------
# Diagram: method spectrum (low → high ambition)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 3.4))
ax.axis("off")
methods = [
    ("Constant\nextrapolation", "#fff3e0"),
    ("Linear-Z\n(last 50 pts)", "#ffe0b8"),
    ("DTW\nsubseq match", "#d8e8d4"),
    ("Tikhonov GN\ninversion", "#a8d5a0"),
    ("Particle /\nKalman", "#7cc070"),
    ("MDN multi-\nmodal (Ambrus)", "#52ad58"),
    ("Hybrid:\ninv. + ML", "#2a8847"),
]
for i, (label, color) in enumerate(methods):
    x = 0.02 + i * 0.135
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, 0.30), 0.115, 0.42, boxstyle="round,pad=0.02",
        fc=color, ec="#2b4a7a", lw=1.0,
    ))
    ax.text(x + 0.057, 0.51, label, ha="center", va="center", fontsize=8.5)
ax.annotate(
    "", xy=(0.99, 0.18), xytext=(0.01, 0.18),
    arrowprops=dict(arrowstyle="->", color="#1f2933", lw=1.4),
)
ax.text(0.01, 0.06, "trivial baseline", fontsize=9, color="#6b7280")
ax.text(0.99, 0.06, "state-of-the-art", fontsize=9, ha="right", color="#6b7280")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save_fig("methods", fig)


# ---------------------------------------------------------------------------
# Diagram: sequential MDN interpretation (mirrors Ambrus poster Fig 1)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(10, 4.5))
md = np.linspace(0, 100, 200)
true_svd = 2 * np.sin(0.06 * md) + 0.04 * md - 2
# N stochastic candidates branching out ahead of "bit"
bit_md = 60
ax.axvline(bit_md, color="#d05a5a", ls="--", lw=1.4, label="bit position")
mask_drilled = md <= bit_md
ax.plot(md[mask_drilled], true_svd[mask_drilled], color="black", lw=2.2, label="drilled (known)")
N = 8
weights = np.linspace(0.3, 0.95, N)
for k, w in enumerate(weights):
    # Candidate diverges ahead of bit
    cand = true_svd.copy()
    drift_const = 0.6 * rng.standard_normal()
    drift_slope = 0.015 * rng.standard_normal()
    bend = 0.8 * np.sin(0.03 * (md - bit_md) + rng.standard_normal())
    cand[~mask_drilled] = (
        true_svd[~mask_drilled]
        + drift_const
        + drift_slope * (md[~mask_drilled] - bit_md)
        + bend[~mask_drilled]
    )
    ax.plot(md[~mask_drilled], cand[~mask_drilled], color=plt.cm.viridis(w),
            lw=1.0, alpha=0.55 + 0.4 * w, label=f"mode {k+1} (p={w:.2f})" if k < 3 else None)
ax.set_xlabel("MD (ft)")
ax.set_ylabel("Stratigraphic Vertical Depth (= TVT)")
ax.set_title("Multi-modal probabilistic inversion: N candidate SVD curves ahead of the bit "
             "(after Ambrus et al. 2022, NORCE)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.25)
save_fig("mdn_concept", fig)


# ---------------------------------------------------------------------------
# Diagram: DTW alignment sketch
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1)
n = 400
z = np.linspace(0, 10, n)
ref = 100 + 25 * (np.sin(0.7 * z) + 0.4 * np.sin(2.3 * z) + 0.2 * rng.standard_normal(n))
warp = 0.4 * np.sin(0.3 * z)  # nonlinear warp
warped_idx = np.clip(np.arange(n) - (warp * 8).astype(int), 0, n - 1)
qry = ref[warped_idx] + 6 * rng.standard_normal(n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
ax = axes[0]
ax.plot(z, ref, color="black", lw=1.2, label="typewell GR")
ax.plot(z, qry, color="#3aa856", lw=1.0, alpha=0.85, label="horizontal-well GR (after PS)")
ax.set_xlabel("depth (typewell) / MD (horizontal)")
ax.set_ylabel("GR")
ax.set_title("Two GR sequences — same geology, slight nonlinear warp")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

ax = axes[1]
# Draw tie lines between matched points (simulate DTW alignment)
for i in range(0, n, 20):
    j = warped_idx[i]
    ax.plot([z[i], z[j]], [100, 0], color="#7b5ec0", lw=0.5, alpha=0.5)
ax.plot(z, np.full_like(z, 100), color="black", lw=1.4)
ax.plot(z, np.zeros_like(z), color="#3aa856", lw=1.4)
ax.text(z[-1], 102, "typewell", ha="right", fontsize=9)
ax.text(z[-1], -3, "horizontal (warped)", ha="right", fontsize=9, color="#3aa856")
ax.set_ylim(-15, 115)
ax.set_xlabel("depth / MD")
ax.set_yticks([])
ax.set_title("DTW alignment ties matched samples across the two sequences")
save_fig("dtw_concept", fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "", width: str = "100%") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:{width};max-width:100%'/></figure>"
    )


pdf_link = ""
if (REFS_DIR / "ambrus_inversion_poster.pdf").exists():
    pdf_link = (
        f" &middot; <a href='refs/ambrus_inversion_poster.pdf'>"
        f"PDF (local copy)</a>"
    )


html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII Wellbore Geology Prediction — Consolidated Knowledge Base</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    background: #fafbfc; color: #1f2933;
    max-width: 1140px; margin: 0 auto; padding: 28px 36px 80px;
    line-height: 1.6;
  }}
  h1 {{ font-size: 32px; border-bottom: 2px solid #3b6fbb; padding-bottom: 8px; margin-top: 0; }}
  h2 {{ font-size: 22px; color: #2b4a7a; margin-top: 38px;
        border-left: 4px solid #3b6fbb; padding-left: 10px; }}
  h3 {{ font-size: 16px; color: #444; margin-top: 24px; }}
  p, li {{ font-size: 14.5px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 18px; }}
  code {{ background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  pre {{ background: #1f2933; color: #f3f4f6; padding: 14px 16px; border-radius: 6px;
         overflow-x: auto; font-size: 12.5px; line-height: 1.5; }}
  blockquote {{ border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
                background: white; color: #2f4d36; font-size: 13.5px; }}
  blockquote.warn {{ border-left-color: #d05a5a; color: #5c2929; background: #fff5f5; }}
  blockquote.idea {{ border-left-color: #b8860b; color: #5c4515; background: #fff9e6; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 18px 0; }}
  .kpi {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }}
  .kpi .v {{ font-size: 20px; font-weight: 600; color: #1f2933; }}
  .kpi .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
                 gap: 16px; margin: 16px 0; }}
  .card {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 14px 16px; }}
  .card h3 {{ margin-top: 0; color: #2b4a7a; font-size: 15px; }}
  .card.featured {{ background: linear-gradient(180deg,#eaf3ff 0%, #ffffff 100%);
                     border-color: #b8d4ec; }}
  .pill {{ display: inline-block; background: #e2efff; color: #2b4a7a; padding: 1px 8px;
           border-radius: 8px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
           text-transform: uppercase; }}
  .pill.warn {{ background: #ffe2e2; color: #8c2929; }}
  .pill.idea {{ background: #fff5cc; color: #6b4f00; }}
  .pill.tool {{ background: #e3f3e0; color: #1f5f1f; }}
  .pill.lit  {{ background: #efe6f7; color: #4f2c80; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 13px;
                margin: 10px 0 18px; background: white; }}
  table.data th, table.data td {{ border-bottom: 1px solid #e3e7ec;
                padding: 7px 10px; text-align: left; vertical-align: top; }}
  table.data th {{ background: #eef3f9; color: #2b4a7a; }}
  a {{ color: #2b6cb0; }}
  hr {{ border: none; border-top: 1px solid #e3e7ec; margin: 30px 0; }}
  .nav {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 24px; }}
  .nav a {{ background: #2b6cb0; color: white; text-decoration: none; padding: 8px 14px;
            border-radius: 6px; font-size: 13px; font-weight: 600; }}
  .nav a.secondary {{ background: #5cb874; }}
  .nav a.tert {{ background: #6c757d; }}
  .toc {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px;
          padding: 12px 18px; margin: 16px 0 24px; font-size: 13.5px; }}
  .toc ol {{ margin: 6px 0; padding-left: 20px; }}
  .toc li {{ margin: 2px 0; }}
</style>
</head>
<body>

<h1>ROGII Wellbore Geology Prediction &mdash; Consolidated Knowledge Base</h1>
<div class="meta">
  All-in-one English knowledge base for
  <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
    Kaggle &middot; rogii-wellbore-geology-prediction</a>.
  Regenerated by <code>report/generate_index.py</code>.
</div>

<div class="nav">
  <a href="eda_report.html">📊 Deep EDA</a>
  <a href="competition_overview.html" class="secondary">🧭 Inverse-Problem Formulation</a>
  <a href="{REPO_URL}" class="tert">🛠 Viewer on GitHub</a>
</div>

<div class="toc">
<b>Contents</b>
<ol>
  <li><a href="#tldr">One-paragraph summary</a></li>
  <li><a href="#kpi">Headline numbers</a></li>
  <li><a href="#snapshot">Dataset snapshot (4-panel)</a></li>
  <li><a href="#insights">Six domain insights</a></li>
  <li><a href="#math">Formal inverse-problem formulation</a></li>
  <li><a href="#methods">Method portfolio</a></li>
  <li><a href="#refs">Reference implementations &amp; literature</a></li>
  <li><a href="#test">Test set in the training landscape</a></li>
  <li><a href="#pitfalls">Pipeline pitfalls</a></li>
  <li><a href="#roadmap">Recommended roadmap</a></li>
  <li><a href="#tooling">Tooling we built</a></li>
  <li><a href="#sources">All sources</a></li>
</ol>
</div>

<h2 id="tldr">1. One-Paragraph Summary</h2>
<blockquote>
This Kaggle competition asks us to predict <code>TVT</code> (true vertical thickness — the
geological-depth coordinate) at every measured-depth (MD) point of a horizontal wellbore beyond a
known <i>Prediction Start (PS)</i> point. Inputs available at scoring time: the well&apos;s 3D
trajectory (<code>X, Y, Z, MD</code>), a noisy gamma-ray log (<code>GR</code>, ~{median_gr_missing:.0f}%
missing on average), the known TVT before PS, and a paired vertical reference log (<i>typewell</i>) of
GR vs TVT. The task is most naturally framed as a <b>geophysical inverse problem</b>: given the
observed GR along the lateral and a typewell GR–depth dictionary, recover the wellbore&apos;s
geological position. Naive baselines reach median per-well RMSE ≈ {med_rmse_const:.1f}&nbsp;ft;
competitive approaches combine signal alignment (DTW), Tikhonov-regularized GR inversion, sequential
filtering, and ML sequence models with auxiliary formation-top supervision. State-of-the-art
literature (Ambrus et al. 2022, NORCE) uses a Mixture Density Network for direct multi-modal
probabilistic inversion at 443&nbsp;ms / step.
</blockquote>

<h2 id="kpi">2. Headline Numbers</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Training wells</div><div class="v">{n_train}</div></div>
  <div class="kpi"><div class="l">Training rows</div><div class="v">{total_train_rows:,}</div></div>
  <div class="kpi"><div class="l">Visible test wells</div><div class="v">{n_test} (template only)</div></div>
  <div class="kpi"><div class="l">Visible test rows to predict</div><div class="v">{total_eval_rows_test:,}</div></div>
  <div class="kpi"><div class="l">Median % of well to predict</div><div class="v">{median_eval_pct:.1f}%</div></div>
  <div class="kpi"><div class="l">Median TVT range / well</div><div class="v">{median_tvt_range:,.0f}&nbsp;ft</div></div>
  <div class="kpi"><div class="l">Median GR missing</div><div class="v">{median_gr_missing:.1f}%</div></div>
  <div class="kpi"><div class="l">Median |dTVT/dMD|</div><div class="v">{median_slope:.3f}</div></div>
  <div class="kpi"><div class="l">Constant baseline RMSE</div><div class="v">{med_rmse_const:,.1f}&nbsp;ft</div></div>
  <div class="kpi"><div class="l">Linear-Z baseline RMSE</div><div class="v">{med_rmse_lin_z:,.1f}&nbsp;ft</div></div>
</div>

<h2 id="snapshot">3. Dataset Snapshot</h2>
{img("snapshot", "dataset snapshot")}
<p>
The dataset is uniform-1-ft-MD-sampled with identical schema across all 773 training wells. Median
well asks us to extend ~{median_eval_pct:.0f}% of its length forward from PS. Geological dips are
mild on average but the tail is steep — any predictor that assumes monotonic or zero dip will fail
on a sizable fraction of wells.
</p>

<h2 id="insights">4. Six Domain Insights</h2>

<div class="card-grid">

<div class="card featured">
<span class="pill">Insight 1</span>
<h3>This is an inverse problem, not a regression task</h3>
<p>Reframe as: recover wellbore position <i>h(MD) = TVT</i> by inverting the forward
operator <i>F(h) := GR_v(h(MD))</i> against observed horizontal GR. Tikhonov / Gauss–Newton template
solves this in &lt; 200 lines of NumPy (see Reference §7).</p>
</div>

<div class="card featured">
<span class="pill">Insight 2</span>
<h3>Shift &amp; stretch = the geosteering question</h3>
<p>A horizontal-well GR snippet matches the typewell GR at the TVT range it was drilled in, modulo a
small affine warp for local dip. This is the geosteering engineer&apos;s mental algorithm and
algorithmically is <b>subsequence DTW / cross-correlation</b>.</p>
</div>

<div class="card featured">
<span class="pill">Insight 3</span>
<h3>Nearby wells&apos; typewells are offset copies of the same log</h3>
<p>773 typewells share only 28 distinct geology labels, dominated by the same 6 formations.
Neighbouring typewells are the same vertical GR pattern shifted in depth. <b>Stack the K nearest
typewells</b> to get a cleaner reference signal.</p>
</div>

<div class="card">
<span class="pill">Insight 4</span>
<h3>Sequential inversion: march one MD point at a time</h3>
<p>The standard LWD workflow reuses the previous MD point&apos;s recovered TVT as the initial guess for
the next. Maps to Kalman / particle filter / sequential MDN inference.</p>
</div>

<div class="card">
<span class="pill">Insight 5</span>
<h3>Formation-top columns are training-only ground truth</h3>
<p>Columns <code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code> hold the exact elevation of each of
the six formation tops at each MD point. Training-only. Use as <b>auxiliary multi-task supervision</b>
in any ML model.</p>
</div>

<div class="card">
<span class="pill">Insight 6</span>
<h3>Test set is hidden — code generalizes, not memorizes</h3>
<p>The 3 wells under <code>test/</code> are a submission-shape template. The real evaluation runs on
a hidden set on Kaggle&apos;s servers. Implication: never hard-code well IDs; load <code>test/</code>
by directory scan; trust leave-one-well-out CV over the public LB.</p>
</div>

</div>

<h2 id="math">5. Formal Inverse-Problem Formulation</h2>
<p>Tikhonov-regularized minimization (Schumann, 2021 — see §7):</p>
<p style="text-align:center; font-family: 'Cambria Math', 'Times New Roman', serif; font-size:15px;">
  min<sub>h</sub> Φ(h) = ‖W<sub>d</sub>(F(h) − d<sub>obs</sub>)‖₂² + β‖W<sub>m</sub> h‖₂²
</p>
<p>The Gauss–Newton normal equation for the per-iteration model update Δh is</p>
<p style="text-align:center; font-family: 'Cambria Math', 'Times New Roman', serif; font-size:15px;">
  (J<sup>T</sup> W<sub>d</sub><sup>T</sup> W<sub>d</sub> J + β W<sub>m</sub><sup>T</sup> W<sub>m</sub>) Δh = −∇Φ
</p>
<p>where J = ∂F/∂h is the Jacobian, W<sub>d</sub> down-weights noisy / missing GR rows, and W<sub>m</sub>
is a finite-difference operator that penalizes TVT-path curvature. β is picked via L-curve criterion
or cross-validation. <b>The full derivation with figures lives in <a href="competition_overview.html">
competition_overview.html</a> §5.</b></p>

<h2 id="methods">6. Method Portfolio</h2>
{img("methods", "method spectrum")}

<table class="data">
  <thead><tr><th>Method</th><th>Verdict</th><th>Use it for</th></tr></thead>
  <tbody>
    <tr><td><span class="pill">Constant</span></td>
        <td>Floor — median RMSE ≈ {med_rmse_const:.1f}&nbsp;ft</td>
        <td>First sanity-check submission.</td></tr>
    <tr><td><span class="pill">Linear-Z</span></td>
        <td>Cuts RMSE roughly in half — median ≈ {med_rmse_lin_z:.1f}&nbsp;ft</td>
        <td>Quick second-pass; reuses lateral&apos;s own geometry.</td></tr>
    <tr><td><span class="pill idea">DTW subseq match</span></td>
        <td>Strong classical baseline; interpretable; no training.</td>
        <td>First &ldquo;real&rdquo; submission. Especially with stacked typewells.</td></tr>
    <tr><td><span class="pill idea">Tikhonov Gauss–Newton</span></td>
        <td>Mathematically clean; handles missing GR.</td>
        <td>Needs a good initialization — use DTW seed or previous-point solution.</td></tr>
    <tr><td><span class="pill idea">Particle / Kalman</span></td>
        <td>Robust to multimodal posteriors; online.</td>
        <td>When linearized inversion exposes local-minimum problems.</td></tr>
    <tr><td><span class="pill lit">MDN sequential</span></td>
        <td>State-of-the-art (Ambrus et al. 2022); 443&nbsp;ms / step.</td>
        <td>If you have GPU + time: tracks N candidate SVD curves with explicit
            probabilities. See Reference §7.</td></tr>
    <tr><td><span class="pill">Sequence model</span></td>
        <td>Powerful but 773 wells is small without strong priors.</td>
        <td>Train with auxiliary heads on the 6 formation-top columns.</td></tr>
    <tr><td><span class="pill idea">Hybrid (inv. → ML residual)</span></td>
        <td>Combines physical fidelity with pattern flexibility.</td>
        <td>Best ceiling; two-stage but stable.</td></tr>
  </tbody>
</table>

<h2 id="refs">7. Reference Implementations &amp; Literature</h2>

<div class="card-grid">

<div class="card">
<span class="pill lit">classical inversion</span>
<h3>LWD_inversion — Tikhonov GR inversion</h3>
<p><a href="https://github.com/hhschumann/LWD_inversion">github.com/hhschumann/LWD_inversion</a>
(Schumann, 2021). ~200 LOC Python/Jupyter. Implements eq. (1)–(6) (in
<a href="competition_overview.html">competition_overview.html</a>) with L-curve regularization.
Forward model: nearest-neighbor lookup on a pilot/typewell GR. Known weakness:
local-minimum trap from linearization → mitigate with sequential initialization.</p>
</div>

<div class="card">
<span class="pill idea">subsequence alignment</span>
<h3>DTW on GR logs — <code>dtw-python</code> recipe</h3>
<p>The classical signal-alignment baseline. From <code>notebook/DTW.ipynb</code> (gitignored
third-party reference):</p>
<pre><code>!pip install dtw-python
from dtw import dtw, rabinerJuangStepPattern

w1 = horizontal_well_gr   # numpy array (post-PS GR)
w2 = typewell_gr          # numpy array (vertical reference GR)
alignment = dtw(
    w1, w2,
    keep_internals=True,
    step_pattern=rabinerJuangStepPattern(1, "c"),
)
alignment.plot(type="twoway", offset=-200)</code></pre>
<p>Directly applicable: feed in <code>HW.GR (post-PS)</code> against the typewell (or a
super-typewell — see Insight 3); read off TVT by indexing the typewell by the alignment indices.</p>
</div>

<div class="card featured">
<span class="pill lit">state-of-the-art</span>
<h3>Ambrus et al. 2022 — sequential MDN multi-modal inversion</h3>
<p><i>AI-based multi-modal interpretation of logs for ahead-of-bit probabilistic ROP prediction.</i>
Presented at NFES / NORCE Geosteering &amp; Formation Evaluation Workshop 2022{pdf_link}.</p>
<ul>
  <li><b>Model:</b> Mixture Density Network outputting <i>N</i> SVD (= TVT) candidates +
      probability per candidate.</li>
  <li><b>Training:</b> 28 M synthetic SVD samples with variable dips and faulting;
      loss = α · I_class + I_MAE (cross-entropy on closest mode + L1 on closest mode).</li>
  <li><b>Sequential interpretation:</b> previous step&apos;s candidates seed the next step;
      coinciding points merge with summed probability; low-probability realizations are dropped.</li>
  <li><b>Speed:</b> 443&nbsp;ms / step on standard CPU; ~100× faster than classical
      stratigraphy interpretation.</li>
  <li><b>Validation:</b> Geosteering World Cup 2020 semi-final well (Middle Woodford fm,
      South Central Oklahoma).</li>
</ul>
{img("mdn_concept", "MDN concept")}
<p style="font-size:12px; color:#6b7280; margin-top:-6px;">
  Sketch: at the bit position, N candidate SVD curves branch out ahead, each with its own
  probability. The classical DTW gives one path; the MDN gives a distribution.</p>
</div>

<div class="card">
<span class="pill lit">cited papers</span>
<h3>Underlying literature</h3>
<ul>
  <li>Alyaev &amp; Elsheikh (2022). <i>Direct multi-modal inversion of geophysical logs using deep
      learning.</i> Earth and Space Science 9(9).</li>
  <li>Alyaev, Ambrus, Jahani &amp; Elsheikh (2022). <i>Sequential Multi-Realization Probabilistic
      Interpretation of Well Logs and Geological Prediction by a Deep-Learning Method.</i> SPWLA
      Annual Logging Symposium.</li>
  <li>Tadjer, Alyaev, Miner, Kuvaev, Bratvold (2021). <i>Unlocking the Human Factor: Geosteering
      Decision Making as a Component of Drilling Operational Efficacy.</i> SPE/AAPG/SEG
      Unconventional Resources Technology Conference, July 27, 2021.</li>
</ul>
</div>

<div class="card">
<span class="pill tool">production reference</span>
<h3>ROGII StarSteer — Auto Steer</h3>
<p><a href="https://www.rogii.com/">rogii.com</a> &middot; commercial geosteering software. Its
&ldquo;Auto Steer&rdquo; module outputs exactly the TVD-vs-MD cross-section + per-MD confidence the
competition is asking for. The competition is effectively a public benchmark of that
auto-interpretation task.</p>
</div>

</div>

<h3>DTW intuition — visual</h3>
{img("dtw_concept", "DTW concept")}
<p>DTW (dynamic time warping) finds a monotone many-to-one tie between two sequences that minimizes
the cumulative point-to-point distance. For GR logs, that means matching geological signatures even
when one sequence is locally stretched or compressed relative to the other (= local dip).</p>

<h2 id="test">8. Test Set in the Training Landscape</h2>
{img("map", "test wells in train context")}
<p>The 3 visible test wells fall inside the training XY distribution; their 5 nearest training
neighbours by centroid distance:</p>
<table class="data">
  <thead><tr><th>Test well</th><th>Rank</th><th>Nearest train well</th><th>Distance (ft)</th></tr></thead>
  <tbody>
""" + "\n".join(
    f"    <tr><td>{r.test_well}</td><td>{r.rank}</td><td><code>{r.nearest_train_well}</code></td>"
    f"<td>{r.distance_ft:,.0f}</td></tr>"
    for r in nn.itertuples()
) + """
  </tbody>
</table>
<p>These are excellent candidates for offset-well priors and for the "stack the K nearest typewells"
trick (Insight 3).</p>

<h2 id="pitfalls">9. Pipeline Constraints &amp; Pitfalls</h2>
<ul>
  <li><b>Hidden test:</b> code must be well-id agnostic — scan the directory.</li>
  <li><b>CV:</b> Leave-One-Well-Out or GroupKFold by XY cluster. Random row splits leak.</li>
  <li><b>GR missingness</b> is <i>blocky</i> in many wells — use a mask, not mean-imputation.</li>
  <li><b>Don&apos;t leak the formation-top columns at inference</b> — they are training-only.</li>
  <li><b>TVT ≠ TVD.</b> The Z column is surface-relative elevation; TVT is the typewell&apos;s
      vertical coordinate. They differ by a per-region datum offset.</li>
  <li><b>Direction is not monotone.</b> A well&apos;s TVT can increase, decrease, or stay flat within
      a single lateral — any model that assumes monotonic TVT will fail on the steep tail.</li>
  <li><b>DTW caveat:</b> long sequences are O(N·M) in memory; for our 4000 × 2000 typical sizes
      use a Sakoe-Chiba band (constrained DTW) or down-sample by 4–8× before alignment.</li>
</ul>

<h2 id="roadmap">10. Recommended Roadmap</h2>
<ol>
  <li>Reproduce <code>report/baseline_rmse.csv</code> → ship a constant-baseline
      <code>submission.csv</code> to confirm the format is correct.</li>
  <li>Build a DTW predictor over the assigned typewell → first credible submission.</li>
  <li>Stack 5–10 nearest typewells into a "super-typewell" → re-run DTW.</li>
  <li>Implement Tikhonov / Gauss–Newton inversion (see
      <a href="competition_overview.html">competition_overview.html</a> §5).
      Initialize from DTW seed.</li>
  <li>Wrap in a sequential filter (particle or unscented Kalman) for online stability.</li>
  <li>Train a 1-D CNN / Transformer with formation-top heads as auxiliary supervision.</li>
  <li><i>Stretch goal:</i> reproduce the Ambrus MDN approach — synthetic SVD pre-training +
      sequential candidate tracking. State-of-the-art per the 2022 NFES poster.</li>
  <li>Ensemble: classical inversion + ML; let ML learn the residual.</li>
</ol>

<h2 id="tooling">11. Tooling We Built</h2>
<div class="card-grid">

<div class="card">
<span class="pill tool">desktop app</span>
<h3>ROGII Viewer</h3>
<p>PySide6 + pyqtgraph geosteering viewer. Loads any folder of well CSVs, renders a cross-section
with formation-top bands, GR correlation, TVT prediction overlay, and map view. <b>Loads any
Kaggle-format predictions CSV and shows per-well RMSE vs truth.</b></p>
<p>Source: <a href="{REPO_URL}">{REPO_URL}</a> &middot; Linux/WSL launcher:
<code>bash viewer/run.sh</code> &middot; Windows .exe via <code>viewer/build.bat</code>.</p>
</div>

<div class="card">
<span class="pill tool">report</span>
<h3>Deep EDA</h3>
<p>14 figures across 773 wells, baseline RMSE floors, missingness analysis, formation-dip stats,
geology vocabulary. <a href="eda_report.html">eda_report.html</a></p>
</div>

<div class="card">
<span class="pill tool">report</span>
<h3>Problem &amp; Methods Overview</h3>
<p>Formal inverse-problem framing with full Tikhonov / Gauss–Newton math, method portfolio.
<a href="competition_overview.html">competition_overview.html</a></p>
</div>

</div>

<h2 id="sources">12. All Sources</h2>
<ul>
  <li><b>Competition:</b>
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
        kaggle.com/competitions/rogii-wellbore-geology-prediction</a></li>
  <li><b>Community discussions:</b>
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431">697431</a>
      · <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853">699853</a>
      · <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326">699326</a></li>
  <li><b>Classical inversion reference (Tikhonov + GN):</b>
      <a href="https://github.com/hhschumann/LWD_inversion">github.com/hhschumann/LWD_inversion</a>
      (Schumann, 2021)</li>
  <li><b>DTW reference notebook:</b> <code>notebook/DTW.ipynb</code> (two-well GR alignment with
      <code>dtw-python</code>).</li>
  <li><b>Multi-modal MDN inversion:</b> Ambrus, Alyaev, Jahani, Elsheikh (2022)
      &mdash; NFES/NORCE Workshop poster{pdf_link}.</li>
  <li><b>Underlying papers:</b> Alyaev &amp; Elsheikh (2022, Earth &amp; Space Science 9(9)) ·
      Alyaev et al. (2022, SPWLA) · Tadjer et al. (2021, URTC).</li>
  <li><b>Production reference:</b>
      <a href="https://www.rogii.com/">ROGII StarSteer Auto Steer</a></li>
  <li><b>Top {len(top_geology)} typewell geology labels in the dataset:</b>
      {", ".join(f"<code>{g}</code>" for g in top_geology)}</li>
</ul>

<hr/>
<p class="meta">
This file is the entry point for the consolidated knowledge base. For the full data EDA see
<a href="eda_report.html">eda_report.html</a>; for the problem framing + math see
<a href="competition_overview.html">competition_overview.html</a>; for the desktop viewer source
see <a href="{REPO_URL}">{REPO_URL}</a>. Last regenerated by <code>report/generate_index.py</code>.
</p>

</body>
</html>
"""

out = REPORT_DIR / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
