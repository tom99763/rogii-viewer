"""Generate report/wells_on_geology.html — wells drawn on top of the geology.

For every training horizontal well, this report shows:
  * The full XY trajectory (heel-to-toe line) painted on top of geological
    backgrounds (Eagle Ford depth, Eagle Ford thickness, structural dip).
  * The toe (= end of well) as a downward triangle marker.
  * Color encoding by lateral length, mean TVT, and mean GR.
  * A 2D toe-density heatmap (where lateral drilling concentrates).
  * Side-by-side histograms of lateral length / TVT / GR distributions.

The point: see how horizontal drilling relates to the geology of the play.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

DATA_ROOT = Path("/home/tom99763/ROGII/rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
REPORT_DIR = Path("/home/tom99763/ROGII/report")

FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 1. Load formation-top observations (for the geology background grids)
#    and full trajectories (for the wellbore overlays)
# ---------------------------------------------------------------------------
print("loading well trajectories + formation tops...")
t0 = time.time()
cols = ["MD", "X", "Y", "Z", "GR", "TVT", *FORMATIONS]
hw_paths = sorted(
    p for p in TRAIN_DIR.glob("*__horizontal_well.csv") if "Zone" not in p.name
)

# For background grid: every 40th row across all wells
bg_parts = []
# For overlays: every well's full XY trajectory (subsampled every 25 rows)
traj_xy: list[np.ndarray] = []
heels = []
toes = []
well_stats = []

for i, p in enumerate(hw_paths):
    df = pd.read_csv(p, usecols=cols)
    bg_parts.append(df[["X", "Y", *FORMATIONS]].iloc[::40].copy())
    sub = df.iloc[::25]
    xy = sub[["X", "Y"]].to_numpy()
    traj_xy.append(xy)
    heels.append(xy[0])
    toes.append(xy[-1])
    # Per-well stats for color encoding
    md_span = float(df["MD"].max() - df["MD"].min())
    well_stats.append({
        "well_id": p.name.split("__", 1)[0],
        "heel_x": xy[0, 0], "heel_y": xy[0, 1],
        "toe_x": xy[-1, 0], "toe_y": xy[-1, 1],
        "lateral_len_ft": md_span,
        "tvt_mean": float(df["TVT"].mean()),
        "tvt_min": float(df["TVT"].min()),
        "gr_mean": float(df["GR"].mean(skipna=True)),
    })
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(hw_paths)}  ({time.time()-t0:.1f}s)")

points = pd.concat(bg_parts, ignore_index=True)
ws = pd.DataFrame(well_stats)
heels = np.array(heels)
toes = np.array(toes)
print(f"  -> {len(points):,} background obs, {len(traj_xy)} trajectories in {time.time()-t0:.1f}s")

# Test wells (we want to overlay their trajectories too)
test_paths = sorted(
    p for p in TEST_DIR.glob("*__horizontal_well.csv") if "Zone" not in p.name
)
test_traj = []
test_stats = []
for p in test_paths:
    df = pd.read_csv(p, usecols=["MD", "X", "Y"])
    sub = df.iloc[::25]
    test_traj.append(sub[["X", "Y"]].to_numpy())
    test_stats.append({
        "well_id": p.name.split("__", 1)[0],
        "lateral_len_ft": float(df["MD"].max() - df["MD"].min()),
    })


# ---------------------------------------------------------------------------
# 2. Build interpolated geology grids on a common XY mesh
# ---------------------------------------------------------------------------
x_min, x_max = points["X"].min(), points["X"].max()
y_min, y_max = points["Y"].min(), points["Y"].max()
mx, my = 0.02 * (x_max - x_min), 0.02 * (y_max - y_min)
x_min -= mx; x_max += mx; y_min -= my; y_max += my
nx, ny = 140, 110
xi = np.linspace(x_min, x_max, nx)
yi = np.linspace(y_min, y_max, ny)
XX, YY = np.meshgrid(xi, yi)
xy_pts = points[["X", "Y"]].to_numpy()

print("interpolating geology grids...")
grids = {}
for fm in ("EGFDU", "BUDA"):
    Z = griddata(xy_pts, points[fm].to_numpy(), (XX, YY), method="linear")
    mask = ~np.isnan(Z)
    Zs = Z.copy()
    Zs[mask] = gaussian_filter(np.where(mask, Z, 0), sigma=1.2)[mask]
    grids[fm] = Zs

eagle_top = grids["EGFDU"]
eagle_thick = grids["EGFDU"] - grids["BUDA"]

# Dip from EGFDU
dy_step = YY[1, 0] - YY[0, 0]
dx_step = XX[0, 1] - XX[0, 0]
gz_dy, gz_dx = np.gradient(eagle_top, dy_step, dx_step)
dip_mag = np.hypot(gz_dx, gz_dy)


# ---------------------------------------------------------------------------
# 3. Headline 2×2: wells on top of geology, colored by lateral length / TVT / GR
# ---------------------------------------------------------------------------
def add_wells(ax, color_vals, cmap, label, vmin=None, vmax=None,
              line_alpha=0.85, line_width=1.0):
    """Draw all 773 training trajectories as colored LineCollection."""
    norm = plt.Normalize(vmin if vmin is not None else np.nanpercentile(color_vals, 2),
                         vmax if vmax is not None else np.nanpercentile(color_vals, 98))
    segments = []
    colors = []
    for xy, c in zip(traj_xy, color_vals):
        seg = xy / 1000.0
        segments.append(seg)
        colors.append(cmap(norm(c)))
    lc = LineCollection(segments, colors=colors, linewidths=line_width, alpha=line_alpha)
    ax.add_collection(lc)
    # Toes
    sc = ax.scatter(toes[:, 0] / 1000, toes[:, 1] / 1000,
                    c=color_vals, cmap=cmap, norm=norm,
                    s=8, marker="v", edgecolor="black", linewidth=0.2, zorder=4)
    # Test wells
    for tx in test_traj:
        ax.plot(tx[:, 0] / 1000, tx[:, 1] / 1000,
                color="black", lw=2.0, zorder=6)
        ax.scatter(tx[-1, 0] / 1000, tx[-1, 1] / 1000,
                   s=80, marker="*", c="white", edgecolor="black",
                   linewidth=1.0, zorder=7)
    return sc

fig, axes = plt.subplots(2, 2, figsize=(15.5, 13), constrained_layout=True)
fig.suptitle("773 horizontal wells overlaid on the geology of the play\n"
             "(black lines/stars = the 3 visible test wells)", fontsize=14, weight="bold")

# (a) wells on Eagle Ford depth; color = lateral length
ax = axes[0, 0]
vmin, vmax = np.nanpercentile(eagle_top, [2, 98])
cf = ax.contourf(XX / 1000, YY / 1000, eagle_top, levels=18,
                 cmap="RdYlBu_r", vmin=vmin, vmax=vmax, alpha=0.7)
sc = add_wells(ax, ws["lateral_len_ft"].to_numpy(), plt.cm.viridis,
               "lateral length (ft)")
ax.set_aspect("equal", adjustable="box")
ax.set_title("(a) Wells on Eagle Ford TOP elevation\nWells colored by lateral length (ft)")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
cb1 = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, location="left")
cb1.set_label("EGFDU elevation (ft)", fontsize=9)
cb2 = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cb2.set_label("lateral length (ft)", fontsize=9)

# (b) wells on Eagle Ford thickness; color = mean TVT
ax = axes[0, 1]
vmin, vmax = np.nanpercentile(eagle_thick, [2, 98])
cf = ax.contourf(XX / 1000, YY / 1000, eagle_thick, levels=18,
                 cmap="YlOrRd", vmin=vmin, vmax=vmax, alpha=0.7)
sc = add_wells(ax, ws["tvt_mean"].to_numpy(), plt.cm.coolwarm,
               "mean TVT (ft)")
ax.set_aspect("equal", adjustable="box")
ax.set_title("(b) Wells on Eagle Ford reservoir THICKNESS\nWells colored by mean TVT")
ax.set_xlabel("X (×1000 ft)")
cb1 = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, location="left")
cb1.set_label("Eagle Ford thickness (ft)", fontsize=9)
cb2 = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cb2.set_label("mean TVT (ft)", fontsize=9)

# (c) wells on structural dip; color = mean GR
ax = axes[1, 0]
vmin, vmax = 0, np.nanpercentile(dip_mag * 100, 96)
cf = ax.contourf(XX / 1000, YY / 1000, dip_mag * 100, levels=18,
                 cmap="magma_r", vmin=vmin, vmax=vmax, alpha=0.55)
sc = add_wells(ax, ws["gr_mean"].to_numpy(), plt.cm.cividis,
               "mean GR")
ax.set_aspect("equal", adjustable="box")
ax.set_title("(c) Wells on STRUCTURAL DIP (% slope)\nWells colored by mean GR (API)")
ax.set_xlabel("X (×1000 ft)")
ax.set_ylabel("Y (×1000 ft)")
cb1 = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02, location="left")
cb1.set_label("dip magnitude (%)", fontsize=9)
cb2 = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cb2.set_label("mean GR (API)", fontsize=9)

# (d) toe-density heatmap + heels
ax = axes[1, 1]
H, xedges, yedges = np.histogram2d(
    toes[:, 0] / 1000, toes[:, 1] / 1000,
    bins=[60, 50],
    range=[[x_min / 1000, x_max / 1000], [y_min / 1000, y_max / 1000]],
)
# Light smooth
Hs = gaussian_filter(H.T, sigma=1.5)
im = ax.imshow(Hs, origin="lower",
               extent=[x_min / 1000, x_max / 1000, y_min / 1000, y_max / 1000],
               cmap="hot_r", alpha=0.85, aspect="auto",
               vmin=0, vmax=np.percentile(Hs, 97))
# Overlay all toes (black) and heels (cyan) as small markers
ax.scatter(heels[:, 0] / 1000, heels[:, 1] / 1000,
           s=8, c="cyan", marker="o", edgecolor="black", linewidth=0.2, zorder=4, label="heel")
ax.scatter(toes[:, 0] / 1000, toes[:, 1] / 1000,
           s=8, c="black", marker="v", linewidth=0, zorder=5, label="toe")
ax.set_aspect("equal", adjustable="box")
ax.set_title("(d) Where laterals END\n(toe density heatmap; cyan = heels, black ▾ = toes)")
ax.set_xlabel("X (×1000 ft)")
ax.legend(loc="lower right", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="toes per bin")
save_fig("wells_on_geology", fig)


# ---------------------------------------------------------------------------
# 4. Distribution panels: lateral length, TVT, GR, azimuth rose
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

ax = axes[0, 0]
ax.hist(ws["lateral_len_ft"], bins=40, color="#3b6fbb", edgecolor="white")
med = ws["lateral_len_ft"].median()
ax.axvline(med, color="crimson", ls="--", label=f"median {med:,.0f} ft ({med/5280:.1f} mi)")
for t in test_stats:
    ax.axvline(t["lateral_len_ft"], color="black", ls=":", lw=1.0)
ax.set_title("Lateral length distribution (MD span)\nBlack dotted = visible test wells")
ax.set_xlabel("lateral length (ft)")
ax.set_ylabel("well count")
ax.legend()

ax = axes[0, 1]
ax.hist(ws["tvt_mean"], bins=40, color="#5cb874", edgecolor="white")
ax.axvline(ws["tvt_mean"].median(), color="crimson", ls="--",
           label=f"median {ws['tvt_mean'].median():,.0f} ft")
ax.set_title("Mean TVT per well (drilling depth)")
ax.set_xlabel("mean TVT (ft)")
ax.set_ylabel("well count")
ax.legend()

ax = axes[1, 0]
ax.hist(ws["gr_mean"].dropna(), bins=40, color="#e9924b", edgecolor="white")
ax.axvline(ws["gr_mean"].median(), color="crimson", ls="--",
           label=f"median {ws['gr_mean'].median():.1f} API")
ax.set_title("Mean GR per well (lithology proxy)")
ax.set_xlabel("mean GR (API)")
ax.set_ylabel("well count")
ax.legend()

# Azimuth rose (from heel→toe vector)
ax = axes[1, 1]
ax.remove()
ax = fig.add_subplot(2, 2, 4, projection="polar")
vec = toes - heels
az = np.arctan2(vec[:, 1], vec[:, 0])  # angle from East, CCW
# Fold to [0, π] to merge ± (azimuth, not direction)
az_folded = np.where(az < 0, az + np.pi, az)
bins = np.linspace(0, np.pi, 19)
counts, _ = np.histogram(az_folded, bins=bins)
theta = (bins[:-1] + bins[1:]) / 2
ax.bar(theta, counts, width=np.diff(bins), bottom=0,
       color="#3b6fbb", edgecolor="white", alpha=0.85)
ax.bar(theta + np.pi, counts, width=np.diff(bins), bottom=0,
       color="#3b6fbb", edgecolor="white", alpha=0.85)
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_title("Lateral azimuth distribution\n(heel→toe direction, 0=East)")
save_fig("distributions", fig)


# ---------------------------------------------------------------------------
# 5. Lateral length vs Eagle Ford thickness (where the wells were placed)
# ---------------------------------------------------------------------------
# Look up Eagle Ford thickness at each well's centroid
def grid_value_at(xs, ys, grid):
    """Bilinear-ish lookup: just nearest cell."""
    out = np.full(len(xs), np.nan)
    for k, (x, y) in enumerate(zip(xs, ys)):
        ix = int(np.clip(round((x - xi[0]) / (xi[-1] - xi[0]) * (nx - 1)), 0, nx - 1))
        iy = int(np.clip(round((y - yi[0]) / (yi[-1] - yi[0]) * (ny - 1)), 0, ny - 1))
        out[k] = grid[iy, ix]
    return out

ws["eagle_top_at_well"] = grid_value_at(ws["heel_x"], ws["heel_y"], eagle_top)
ws["eagle_thick_at_well"] = grid_value_at(ws["heel_x"], ws["heel_y"], eagle_thick)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

ax = axes[0]
ax.scatter(ws["eagle_top_at_well"], ws["tvt_mean"], s=8, alpha=0.5, c="#3b6fbb")
ax.set_xlabel("Eagle Ford TOP elevation at well centroid (ft)")
ax.set_ylabel("Mean TVT of well (ft)")
ax.set_title("Mean TVT vs local Eagle Ford depth\n(wells land in the same zone everywhere)")
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(ws["eagle_thick_at_well"], ws["lateral_len_ft"], s=8, alpha=0.5, c="#e9924b")
ax.set_xlabel("Eagle Ford reservoir thickness at well centroid (ft)")
ax.set_ylabel("Lateral length (ft)")
ax.set_title("Lateral length vs local reservoir thickness")
ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(ws["eagle_thick_at_well"], ws["gr_mean"], s=8, alpha=0.5, c="#5cb874")
ax.set_xlabel("Eagle Ford reservoir thickness at well centroid (ft)")
ax.set_ylabel("Mean GR (API)")
ax.set_title("Mean GR vs local reservoir thickness")
ax.grid(alpha=0.3)
save_fig("scatter_geology", fig)


# ---------------------------------------------------------------------------
# 6. Headline numbers
# ---------------------------------------------------------------------------
med_len = float(ws["lateral_len_ft"].median())
p5_len = float(ws["lateral_len_ft"].quantile(0.05))
p95_len = float(ws["lateral_len_ft"].quantile(0.95))
med_tvt = float(ws["tvt_mean"].median())
med_gr = float(ws["gr_mean"].dropna().median())
median_eagle_thick = float(np.nanmedian(eagle_thick))


# ---------------------------------------------------------------------------
# 7. HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:100%;max-width:100%'/></figure>"
    )


html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII — Wells on Geology</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    background: #fafbfc; color: #1f2933;
    max-width: 1180px; margin: 0 auto; padding: 28px 36px 80px;
    line-height: 1.6;
  }}
  h1 {{ font-size: 30px; border-bottom: 2px solid #3b6fbb; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 22px; color: #2b4a7a; margin-top: 38px;
        border-left: 4px solid #3b6fbb; padding-left: 10px; }}
  p, li {{ font-size: 14.5px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 18px; }}
  code {{ background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }}
  blockquote {{ border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
                background: white; color: #2f4d36; font-size: 13.5px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; margin: 18px 0; }}
  .kpi {{ background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }}
  .kpi .v {{ font-size: 19px; font-weight: 600; color: #1f2933; }}
  .kpi .l {{ font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  figure {{ margin: 10px 0 26px; }}
  a {{ color: #2b6cb0; }}
  hr {{ border: none; border-top: 1px solid #e3e7ec; margin: 30px 0; }}
</style>
</head>
<body>

<h1>ROGII — Wells on the Geology</h1>
<div class="meta">
  Companion to <a href="geological_map.html">geological_map.html</a>: this report
  paints the 773 training horizontal trajectories <i>on top of</i> the
  interpolated geology, and visualises how lateral length, mean TVT and mean
  GR are distributed across the play.
  Generated by <code>report/generate_wells_on_geology.py</code>.
</div>

<h2>1. Headline Numbers</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Lateral length (median)</div>
    <div class="v">{med_len:,.0f} ft ({med_len/5280:.2f} mi)</div></div>
  <div class="kpi"><div class="l">Lateral length p5–p95</div>
    <div class="v">{p5_len:,.0f} – {p95_len:,.0f} ft</div></div>
  <div class="kpi"><div class="l">Mean TVT (median across wells)</div>
    <div class="v">{med_tvt:,.0f} ft</div></div>
  <div class="kpi"><div class="l">Mean GR (median across wells)</div>
    <div class="v">{med_gr:.1f} API</div></div>
  <div class="kpi"><div class="l">Eagle Ford thickness (median)</div>
    <div class="v">{median_eagle_thick:.0f} ft</div></div>
  <div class="kpi"><div class="l">Wells / test wells</div>
    <div class="v">{len(ws)} / {len(test_traj)}</div></div>
</div>

<h2>2. Wellbores on the Geological Background</h2>
{img("wells_on_geology")}
<p>
Four overlays, all on the same XY base map. Each thin colored line is one
training well's horizontal trajectory from heel to toe (downward triangle).
The three black lines with white stars are the visible test wells.
</p>
<ul>
  <li><b>(a) Eagle Ford TOP depth as backdrop.</b> Lateral length encoded in
      well color. Longer wells (yellow-green) are spread across the deeper
      south-central portion of the play; shorter wells cluster more sparsely.</li>
  <li><b>(b) Eagle Ford THICKNESS as backdrop.</b> Mean TVT per well encoded
      in color. The redder/bluer wells track the local Eagle Ford depth — wells
      that drill the deeper Eagle Ford have correspondingly deeper mean TVT.</li>
  <li><b>(c) Structural dip as backdrop.</b> Mean GR encoded in color. Wells
      with higher mean GR (yellow on cividis) cluster in particular sub-regions
      — there's a real spatial pattern in lithology, not just random noise.</li>
  <li><b>(d) Toe density heatmap.</b> Heels (cyan circles) and toes (black
      triangles) are not uniformly distributed — there are well-pad clusters
      where many laterals share a heel and fan out, and broader regions where
      drilling is sparser.</li>
</ul>

<blockquote>
<b>The takeaway:</b> Drilling is structured, not random. Wells avoid the
shallowest and deepest extremes; they target a relatively narrow Eagle Ford
depth window; they often share a pad (so the heel positions are clustered).
Per Insight 3 in our knowledge base, neighbours <i>are</i> physically
relevant — this view makes it visible.
</blockquote>

<h2>3. Distributions</h2>
{img("distributions")}
<p>
Lateral length distribution is right-skewed: most wells are ~5,500–7,500&nbsp;ft
laterals, with a tail of very long wells extending to ~10,000+&nbsp;ft. Mean TVT
is tightly distributed (the field has a coherent target horizon). Lateral
azimuth is bimodal at roughly NE–SW and NW–SE — the classic field-wide
drilling pattern oriented perpendicular to the regional max-horizontal-stress
direction (for optimal fracture-network exposure).
</p>

<h2>4. Lateral Length and Geology Relationships</h2>
{img("scatter_geology")}
<p>
Sampling the interpolated geology at each well's heel gives a per-well
"local geology" feature. Three scatter plots:
</p>
<ul>
  <li><b>Mean TVT vs local Eagle Ford depth</b> — strong negative slope:
      wells in regions where the Eagle Ford is deep have deeper mean TVT.
      This is the "geological coordinate ≈ depth coordinate" relationship that
      a linear-Z baseline exploits.</li>
  <li><b>Lateral length vs local reservoir thickness</b> — relatively flat:
      drillers don't make laterals proportional to reservoir thickness;
      lateral length is mostly driven by lease geometry and pad geometry.</li>
  <li><b>Mean GR vs local reservoir thickness</b> — weak relationship.
      The GR variability dominantly comes from where within the Eagle Ford the
      lateral was steered, not from the bulk thickness available.</li>
</ul>

<h2>5. What This Adds On Top of <code>geological_map.html</code></h2>
<p>
<a href="geological_map.html">geological_map.html</a> showed the geology
without wells; this report adds the actual drilling to that picture. Together:
</p>
<ul>
  <li>The <b>geology surface</b> is interpolated from formation-top observations
      sampled <i>along the wellbores themselves</i> — so the interpolation is
      most reliable where wells are densest (panel d).</li>
  <li>Test wells (white stars in panels a–c, black trajectories in all four)
      fall in well-sampled regions — no extrapolation risk.</li>
  <li>The clear visual structure (parallel laterals, two azimuth clusters,
      clustered pads, single-zone landing) confirms that the dataset comes
      from one coherent operational area and the same operator(s).</li>
</ul>

<hr/>
<p class="meta">
Sibling reports: <a href="eda_report.html">eda_report.html</a> (per-well
statistics) ·
<a href="geological_map.html">geological_map.html</a> (geology only) ·
<a href="competition_overview.html">competition_overview.html</a> (problem
framing + methods) ·
<a href="index.html">index.html</a> (consolidated knowledge base).
</p>

</body>
</html>
"""

out = REPORT_DIR / "wells_on_geology.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
