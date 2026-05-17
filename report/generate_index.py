"""Generate report/index.html — a consolidated knowledge-base front page for the
ROGII Wellbore Geology Prediction work.

Bundles, in one self-contained HTML file:
  * Headline KPIs (computed from the existing CSV artifacts).
  * The 6 domain insights surfaced via community discussions + reference repos.
  * A method portfolio with verdicts.
  * Direct links to the two deep-dive reports (eda_report.html,
    competition_overview.html) and to the GitHub viewer project.

Reads existing artifacts in report/; does not re-run the heavy passes.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path("/home/tom99763/ROGII/report")
REPO_URL = "https://github.com/tom99763/rogii-viewer"

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# Load existing summaries
ws = pd.read_csv(REPORT_DIR / "well_summary.csv")
br = pd.read_csv(REPORT_DIR / "baseline_rmse.csv")
gc = pd.read_csv(REPORT_DIR / "geology_counts.csv")
nn = pd.read_csv(REPORT_DIR / "test_nearest_neighbours.csv")

ts = ws[ws.split == "train"]

# ---------------------------------------------------------------------------
# One mega-figure: at-a-glance dataset snapshot (4 quadrants)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# (a) Per-well evaluation %  +  TVT direction breakdown
ax = axes[0, 0]
ax.hist(ts["eval_pct"], bins=40, color="#3b6fbb", edgecolor="white")
ax.axvline(ts["eval_pct"].median(), color="crimson", ls="--",
           label=f"median {ts['eval_pct'].median():.1f}%")
ax.set_title("(a) % of well that needs prediction\n(eval rows / total rows)")
ax.set_xlabel("eval %")
ax.set_ylabel("well count")
ax.legend()

# (b) baseline RMSE comparison
ax = axes[0, 1]
data = [br["rmse_constant"], br["rmse_linear_md"], br["rmse_linear_z"], br["rmse_oracle_const"]]
labels = ["constant", "linear-MD", "linear-Z", "oracle\nconst"]
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#e2efff", edgecolor="#3b6fbb"),
                medianprops=dict(color="crimson"))
ax.set_yscale("log")
ax.set_ylabel("RMSE (ft, log scale)")
ax.set_title("(b) Baseline RMSE floors\n(simulated on training wells)")

# (c) GR missingness distribution
ax = axes[1, 0]
ax.hist(ts["gr_missing_pct"], bins=40, color="#d05a5a", edgecolor="white")
ax.axvline(ts["gr_missing_pct"].median(), color="black", ls="--",
           label=f"median {ts['gr_missing_pct'].median():.1f}%")
ax.set_title("(c) GR missing rate per well")
ax.set_xlabel("missing %")
ax.set_ylabel("well count")
ax.legend()

# (d) absolute dip distribution
ax = axes[1, 1]
ax.hist(np.clip(ts["slope_abs_med"], 0, 1), bins=40, color="#7b5ec0", edgecolor="white")
ax.axvline(ts["slope_abs_med"].median(), color="black", ls="--",
           label=f"median {ts['slope_abs_med'].median():.3f}")
ax.set_title("(d) Median |dTVT/dMD| per well\n(geological dip steepness)")
ax.set_xlabel("ft of TVT per ft of MD")
ax.set_ylabel("well count")
ax.legend()
save_fig("snapshot", fig)


# Map view test wells in train context
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


# Headline numbers
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
# HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "", width: str = "100%") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:{width};max-width:100%'/></figure>"
    )


html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII Wellbore Geology Prediction — Knowledge Base</title>
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
  h3 {{ font-size: 16px; color: #444; margin-top: 22px; }}
  p, li {{ font-size: 14.5px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 18px; }}
  code {{ background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  blockquote {{ border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
                background: white; color: #2f4d36; font-size: 13.5px; }}
  blockquote.warn {{ border-left-color: #d05a5a; color: #5c2929; background: #fff5f5; }}
  blockquote.idea {{ border-left-color: #b8860b; color: #5c4515; background: #fff9e6; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 18px 0; }}
  .kpi {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }}
  .kpi .v {{ font-size: 20px; font-weight: 600; color: #1f2933; }}
  .kpi .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
</style>
</head>
<body>

<h1>ROGII Wellbore Geology Prediction — Knowledge Base</h1>
<div class="meta">
  Master index for everything we have learned so far about
  <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
    Kaggle &middot; rogii-wellbore-geology-prediction</a>.
  Last regenerated by <code>report/generate_index.py</code>.
</div>

<div class="nav">
  <a href="eda_report.html">📊 Deep EDA Report</a>
  <a href="competition_overview.html" class="secondary">🧭 Problem &amp; Methods</a>
  <a href="{REPO_URL}" class="tert">🛠 Viewer on GitHub</a>
</div>

<h2>1. One-Paragraph Summary</h2>
<blockquote>
This Kaggle competition asks us to predict <code>TVT</code> (true vertical thickness — the geological-depth
coordinate) at every measured-depth (MD) point of a horizontal wellbore beyond a known
<i>Prediction Start (PS)</i> point. Inputs available at scoring time: the well&apos;s 3D trajectory
(<code>X, Y, Z, MD</code>), a noisy gamma-ray log (<code>GR</code>, ~29% missing on average), the known
TVT before PS, and a paired vertical reference log (<i>typewell</i>) of GR vs TVT. The task is most
naturally framed as a <b>geophysical inverse problem</b>: given the observed GR along the lateral and a
typewell GR–depth dictionary, recover the wellbore&apos;s geological position. Naive baselines reach
median per-well RMSE ≈ {med_rmse_const:.1f}&nbsp;ft; competitive approaches combine signal-alignment
(DTW), Tikhonov-regularized GR inversion, sequential filtering, and ML sequence models with auxiliary
formation-top supervision.
</blockquote>

<h2>2. Headline Numbers</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Training wells</div><div class="v">{n_train}</div></div>
  <div class="kpi"><div class="l">Training rows</div><div class="v">{total_train_rows:,}</div></div>
  <div class="kpi"><div class="l">Visible test wells</div><div class="v">{n_test} (template only)</div></div>
  <div class="kpi"><div class="l">Visible test rows to predict</div><div class="v">{total_eval_rows_test:,}</div></div>
  <div class="kpi"><div class="l">Median % of well to predict</div><div class="v">{median_eval_pct:.1f}%</div></div>
  <div class="kpi"><div class="l">Median TVT range / well</div><div class="v">{median_tvt_range:,.0f}&nbsp;ft</div></div>
  <div class="kpi"><div class="l">Median GR missing</div><div class="v">{median_gr_missing:.1f}%</div></div>
  <div class="kpi"><div class="l">Median |dTVT/dMD|</div><div class="v">{median_slope:.3f}</div></div>
  <div class="kpi"><div class="l">Constant baseline RMSE (median)</div><div class="v">{med_rmse_const:,.1f}&nbsp;ft</div></div>
  <div class="kpi"><div class="l">Linear-Z baseline RMSE (median)</div><div class="v">{med_rmse_lin_z:,.1f}&nbsp;ft</div></div>
</div>

<h2>3. Dataset Snapshot</h2>
{img("snapshot", "dataset snapshot")}

<p>
The dataset is uniform-1-ft-MD-sampled and visually well-organized:
each well has exactly two CSVs (horizontal + typewell), the schema is identical across the 773 training
wells, and uniform conventions hold for the prediction-start marker (<code>TVT_input</code> goes NaN at
PS). The geology of the surveyed area is mostly mild dips — median |dTVT/dMD| is just {median_slope:.3f}
— but a long tail of wells contain steep dips that defeat naive extrapolation.
</p>

<h2>4. Six Domain Insights (Community + Reference Repo)</h2>

<div class="card-grid">

<div class="card featured">
<span class="pill">Insight 1</span>
<h3>This is an inverse problem, not a regression task</h3>
<p>Reframe the task as: recover the wellbore position <i>h(MD) = TVT</i> by inverting the forward
operator <i>F(h) := GR_v(h(MD))</i> against the observed horizontal GR. The Tikhonov / Gauss–Newton
template solves this in &lt; 200 LOC. See
<a href="https://github.com/hhschumann/LWD_inversion">github.com/hhschumann/LWD_inversion</a> for a clean
NumPy reference.</p>
</div>

<div class="card featured">
<span class="pill">Insight 2</span>
<h3>Shift &amp; stretch = the geosteering question</h3>
<p>A horizontal-well GR snippet matches the typewell GR at the TVT range it was drilled in, modulo a
small affine warp for local dip. This is the geosteering engineer&apos;s mental algorithm.
Algorithmically it is <b>subsequence DTW / cross-correlation</b> on the GR signal.</p>
</div>

<div class="card featured">
<span class="pill">Insight 3</span>
<h3>Nearby wells&apos; typewells are offset copies of the same log</h3>
<p>The 773 typewells share only 28 distinct geology labels (dominated by the same 6 formations).
Neighbouring wells&apos; typewells are essentially the same vertical GR pattern shifted in depth. <b>Stack
the K nearest typewells</b> to get a much cleaner GR reference.</p>
</div>

<div class="card">
<span class="pill">Insight 4</span>
<h3>Sequential inversion: march one MD point at a time</h3>
<p>The standard LWD workflow reuses the previous MD point&apos;s recovered TVT as the initial guess for
the next. State-space view: latent TVT evolves with drift = local dip; GR observation is a nonlinear
function of the state. Kalman / particle filter variants apply directly.</p>
</div>

<div class="card">
<span class="pill">Insight 5</span>
<h3>Formation-top columns leak the answer in training</h3>
<p>Columns <code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code> hold the absolute elevation of each of the
six formation tops at the trajectory point. Training-only. Use them as <b>auxiliary multi-task
supervision</b> in any ML model — the representations transfer to TVT.</p>
</div>

<div class="card">
<span class="pill">Insight 6</span>
<h3>Test set is hidden — code generalizes, not memorizes</h3>
<p>The 3 wells under <code>test/</code> are a submission-shape template. The real evaluation runs on a
hidden well set on Kaggle&apos;s servers. Implication: never hard-code well IDs; load
<code>test/</code> by directory scan; trust a leave-one-well-out CV over the public LB.</p>
</div>

</div>

<h2>5. Method Portfolio (Verdicts)</h2>

<table class="data">
  <thead><tr><th>Method</th><th>Verdict</th><th>Use it for</th></tr></thead>
  <tbody>
    <tr><td><span class="pill">Constant</span></td>
        <td>Floor (median RMSE ≈ {med_rmse_const:.1f}&nbsp;ft).</td>
        <td>First sanity-check submission; metric baseline.</td></tr>
    <tr><td><span class="pill">Linear in Z</span></td>
        <td>Cuts RMSE roughly in half (median ≈ {med_rmse_lin_z:.1f}&nbsp;ft).</td>
        <td>Quick second-pass submission; reuses the lateral&apos;s own geometry.</td></tr>
    <tr><td><span class="pill idea">DTW + offset-typewell stacking</span></td>
        <td>Strong classical baseline; interpretable; no training.</td>
        <td>Should ship as your first &ldquo;real&rdquo; submission.</td></tr>
    <tr><td><span class="pill idea">Tikhonov Gauss–Newton</span></td>
        <td>Mathematically clean; handles missing GR via row-zeroed weights.</td>
        <td>Needs good initialization (use DTW seed or previous-point solution).</td></tr>
    <tr><td><span class="pill idea">Particle / Kalman filter</span></td>
        <td>Robust to multimodal posteriors and missing observations.</td>
        <td>When inversion exposes local-minimum problems.</td></tr>
    <tr><td><span class="pill">ML sequence model</span></td>
        <td>Powerful, but 773 wells is &ldquo;small&rdquo; without strong priors.</td>
        <td>Always train with auxiliary heads on the 6 formation-top columns.</td></tr>
    <tr><td><span class="pill idea">Hybrid (inversion → ML residual)</span></td>
        <td>Combines physical fidelity with pattern flexibility; the residual is well-behaved.</td>
        <td>Best ceiling; two-stage but stable.</td></tr>
  </tbody>
</table>

<h2>6. Test Set in the Training Landscape</h2>
{img("map", "test wells map")}
<p>The 3 visible test wells fall well inside the training XY distribution; their 5 nearest neighbours
(by centroid distance) give a strong starting point for offset-well priors:</p>

<table class="data">
  <thead><tr><th>Test well</th><th>Rank</th><th>Nearest train well</th><th>Distance (ft)</th></tr></thead>
  <tbody>
""" + "\n".join(
    f"    <tr><td>{r.test_well}</td><td>{r.rank}</td><td><code>{r.nearest_train_well}</code></td>"
    f"<td>{r.distance_ft:,.0f}</td></tr>"
    for r in nn.itertuples()
) + f"""
  </tbody>
</table>

<h2>7. Pipeline Constraints &amp; Pitfalls</h2>
<ul>
  <li><b>Hidden test:</b> code must be <i>well-id agnostic</i> — scan the directory.</li>
  <li><b>CV:</b> Leave-One-Well-Out or GroupKFold by XY cluster. Random row splits leak.</li>
  <li><b>GR missingness</b> is blocky in many wells — use a mask, not mean-imputation.</li>
  <li><b>Don&apos;t leak the formation-top columns at inference</b> — they are training-only.</li>
  <li><b>TVT ≠ TVD.</b> The Z column is surface-relative elevation; TVT is the typewell&apos;s vertical
      coordinate. They differ by a per-region datum offset.</li>
  <li><b>Direction is not monotone.</b> A well&apos;s TVT can increase, decrease, or stay flat within a
      single lateral — any model that assumes monotonic TVT will fail on the steep tail.</li>
</ul>

<h2>8. Recommended Roadmap</h2>
<ol>
  <li>Reproduce <code>report/baseline_rmse.csv</code> &rarr; build a constant-baseline submission.</li>
  <li>Build a DTW predictor over the assigned typewell &rarr; first credible submission.</li>
  <li>Stack 5–10 nearest typewells into a &ldquo;super-typewell&rdquo; &rarr; re-run DTW.</li>
  <li>Implement Tikhonov / Gauss–Newton inversion (see <code>competition_overview.html</code>
      §5 for the formulation). Initialize from DTW.</li>
  <li>Wrap in a sequential filter for online stability.</li>
  <li>Train an ML sequence model with formation-top heads as auxiliary supervision.</li>
  <li>Ensemble inversion + ML; let ML learn the residual.</li>
</ol>

<h2>9. Tooling We Built</h2>

<div class="card-grid">

<div class="card">
<span class="pill tool">desktop app</span>
<h3>ROGII Viewer</h3>
<p>A PySide6 + pyqtgraph geosteering viewer. Loads any folder of well CSVs, renders a cross-section
with formation-top bands, GR correlation, TVT prediction overlay, and map view. <b>Loads any
Kaggle-format predictions CSV and shows per-well RMSE vs truth.</b></p>
<p>Source: <a href="{REPO_URL}">{REPO_URL}</a> &middot; Windows .exe via
<code>viewer/build.bat</code>.</p>
</div>

<div class="card">
<span class="pill tool">report</span>
<h3>Deep EDA</h3>
<p>14 figures, baseline RMSE floors, missingness analysis, formation-dip statistics, geology vocabulary,
companion CSVs. <a href="eda_report.html">eda_report.html</a></p>
</div>

<div class="card">
<span class="pill tool">report</span>
<h3>Problem &amp; Methods Overview</h3>
<p>Formal inverse-problem framing (Tikhonov + Gauss–Newton math), method portfolio,
implementation references. <a href="competition_overview.html">competition_overview.html</a></p>
</div>

</div>

<h2>10. References</h2>
<ul>
  <li><b>Competition:</b>
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
        kaggle.com/competitions/rogii-wellbore-geology-prediction</a></li>
  <li><b>LWD inversion reference (Tikhonov, Gauss–Newton):</b>
      <a href="https://github.com/hhschumann/LWD_inversion">
        github.com/hhschumann/LWD_inversion</a> — Schumann (2021), ~200 LOC NumPy / Jupyter.</li>
  <li><b>Community discussions:</b>
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431">697431</a> ·
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853">699853</a> ·
      <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326">699326</a></li>
  <li><b>Production reference:</b>
      <a href="https://www.rogii.com/">ROGII StarSteer</a> — auto-steer module outputs are the
      effective gold standard the competition is benchmarking against.</li>
  <li><b>Top {len(top_geology)} typewell geology labels in the dataset:</b>
      {", ".join(f"<code>{g}</code>" for g in top_geology)}</li>
</ul>

<hr/>
<p class="meta">
This file is the entry point for the consolidated knowledge base. For the full data EDA see
<a href="eda_report.html">eda_report.html</a>; for the problem framing and method math see
<a href="competition_overview.html">competition_overview.html</a>; for the desktop viewer source see
<a href="{REPO_URL}">the GitHub repo</a>.
</p>

</body>
</html>
"""

out = REPORT_DIR / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
