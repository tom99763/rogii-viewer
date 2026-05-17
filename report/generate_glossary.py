"""Generate report/glossary.html — an illustrated English glossary of the
geosteering / well-logging terms used in the ROGII competition.

Each term gets its own matplotlib-generated diagram (schematic + real-data
example) so a reader unfamiliar with petroleum-engineering vocabulary can pick
up the concepts visually.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

DATA_ROOT = Path("/home/tom99763/ROGII/rogii-wellbore-geology-prediction")
TRAIN_DIR = DATA_ROOT / "train"
REPORT_DIR = Path("/home/tom99763/ROGII/report")

# Pick one real well to anchor the examples
DEMO_WELL = "000d7d20"
demo_hw = pd.read_csv(TRAIN_DIR / f"{DEMO_WELL}__horizontal_well.csv")
demo_tw = pd.read_csv(TRAIN_DIR / f"{DEMO_WELL}__typewell.csv")
demo_ps_idx = int(demo_hw["TVT_input"].isna().to_numpy().argmax())
demo_ps_md = float(demo_hw.loc[demo_ps_idx, "MD"])

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Diagram 1: master picture — surface, typewell, horizontal well, geology
# ---------------------------------------------------------------------------
def draw_master():
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-7.5, 0.5)
    ax.axis("off")

    # Surface
    ax.plot([0, 12], [0, 0], color="#3a7d3a", lw=2.5)
    ax.text(0.2, 0.15, "Surface", fontsize=10, color="#3a7d3a", weight="bold")

    # Geological layers (shallow → deep)
    layer_y = [-1.4, -2.4, -3.3, -3.8, -4.3, -5.0, -6.0]
    layer_names = ["ANCC (Anacacho)", "ASTNU (Austin U)", "ASTNL (Austin L)",
                   "EGFDU (Eagle Ford U)", "EGFDL (Eagle Ford L)", "BUDA"]
    layer_colors = ["#c4d3e6", "#9eb9d6", "#7aa1c5", "#dfb88a", "#c8965d", "#a17a55"]

    # Slight dip
    dip = 0.08
    x_grid = np.linspace(0, 12, 200)
    horizons = []
    for y0, c in zip(layer_y, layer_colors):
        y = y0 + dip * (x_grid - 6)
        horizons.append(y)
    # Fill between
    for i in range(len(horizons) - 1):
        ax.fill_between(x_grid, horizons[i], horizons[i + 1],
                        color=layer_colors[i], alpha=0.85)
        ax.plot(x_grid, horizons[i], color="#666", lw=0.6)
    ax.fill_between(x_grid, horizons[-1], -7.5, color="#7a5a3a", alpha=0.7)
    ax.plot(x_grid, horizons[-1], color="#666", lw=0.6)

    # Layer labels on the right
    for y, name, c in zip(layer_y, layer_names, layer_colors):
        y_rt = y + dip * (12 - 6)
        ax.text(12.15, y_rt - 0.3, name, fontsize=8, color="#444", va="center")

    # Typewell — vertical, drilled at x=2
    tw_x = 2.0
    ax.plot([tw_x, tw_x], [0, -6.5], color="black", lw=3.0, solid_capstyle="round")
    ax.plot([tw_x, tw_x], [0, -6.5], color="white", lw=1.0, ls=(0, (1, 1)))
    ax.text(tw_x - 0.05, -6.8, "TYPEWELL\n(vertical pilot well)",
            ha="center", fontsize=9, color="black", weight="bold")
    # GR log on typewell (sketched as a wiggly curve to the left)
    z_tw = np.linspace(-1, -6, 200)
    gr_tw = 0.4 * np.sin(8 * z_tw) + 0.25 * np.sin(18 * z_tw) + 0.15 * np.random.RandomState(0).standard_normal(200)
    ax.plot(tw_x - 0.5 - 0.4 * gr_tw, z_tw, color="#1a5f9e", lw=1.0)
    ax.text(tw_x - 1.0, -3.5, "GR(TVT)\n← typewell GR\n(the dictionary)",
            ha="right", fontsize=8.5, color="#1a5f9e", weight="bold")
    # Typewell label for vertical axis
    ax.annotate("", xy=(tw_x - 0.15, -6), xytext=(tw_x - 0.15, -1),
                arrowprops=dict(arrowstyle="<->", color="#1a5f9e", lw=1.0))
    ax.text(tw_x - 0.4, -3.5, "TVT", rotation=90, color="#1a5f9e",
            fontsize=10, weight="bold", va="center", ha="center")

    # Horizontal well — heel at x=5, drilled curving down then lateral
    hw_x = np.linspace(5, 11, 200)
    hw_y = -2 + (-2.5) * (1 - np.exp(-(hw_x - 5) * 1.8)) + dip * (hw_x - 6)
    ax.plot(hw_x, hw_y, color="#cc2222", lw=2.8, solid_capstyle="round")
    ax.text(5, -1.8, "HORIZONTAL WELL\n(the one we're predicting)",
            fontsize=9, color="#cc2222", weight="bold")
    # Heel + toe
    ax.scatter([hw_x[0]], [hw_y[0]], s=80, color="#cc2222", zorder=5)
    ax.scatter([hw_x[-1]], [hw_y[-1]], s=140, color="#cc2222", marker="v", zorder=5)
    ax.text(hw_x[0] + 0.1, hw_y[0] + 0.15, "heel", fontsize=8, color="#cc2222")
    ax.text(hw_x[-1] + 0.1, hw_y[-1] - 0.05, "toe", fontsize=8, color="#cc2222")

    # GR on horizontal well (overlaid above)
    gr_hw = 0.45 * np.sin(5 * hw_x) + 0.2 * np.sin(13 * hw_x) + 0.1 * np.random.RandomState(2).standard_normal(200)
    ax.plot(hw_x, hw_y - 0.6 - 0.25 * gr_hw, color="#3aa856", lw=0.9)
    ax.text(8, hw_y[100] - 1.05, "GR(MD)  ← horizontal-well GR (the observation)",
            color="#3aa856", fontsize=8.5, weight="bold")

    # Prediction Start marker
    ps_idx = 60
    ax.axvline(hw_x[ps_idx], ymin=0.08, ymax=0.62, color="#d05a5a", lw=1.4, ls="--")
    ax.annotate("Prediction Start\n(PS)", xy=(hw_x[ps_idx], hw_y[ps_idx] + 0.1),
                xytext=(hw_x[ps_idx] - 0.6, -0.5),
                fontsize=9, color="#d05a5a", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#d05a5a"))
    ax.text(hw_x[ps_idx] / 2 + 2.5, -1.0, "known TVT (TVT_input)",
            ha="center", fontsize=8, color="#444", style="italic")
    ax.text((hw_x[ps_idx] + 11) / 2, -1.0, "predict TVT here →",
            ha="center", fontsize=8, color="#cc2222", style="italic")

    # MD label along well path
    ax.annotate("", xy=(hw_x[100], hw_y[100] - 0.05),
                xytext=(hw_x[0], hw_y[0] + 0.05),
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    ax.text(hw_x[50], hw_y[50] + 0.25, "MD = arc length", fontsize=8, color="#444")

    # TVT side axis on horizontal well
    ax.annotate("", xy=(11.5, hw_y[-1]), xytext=(11.5, 0),
                arrowprops=dict(arrowstyle="<->", color="#1a5f9e", lw=0.8))
    ax.text(11.65, hw_y[-1] / 2, "TVT\n(geological depth\nrelative to typewell)",
            fontsize=8, color="#1a5f9e", weight="bold", va="center")

    ax.set_title("The ROGII task in one picture", fontsize=13, weight="bold", pad=10)
    save_fig("master", fig)


# ---------------------------------------------------------------------------
# Diagram 2: MD vs TVD vs TVT
# ---------------------------------------------------------------------------
def draw_md_tvd_tvt():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # MD
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-7, 0.5)
    ax.plot([0, 10], [0, 0], color="#3a7d3a", lw=2.0)
    ax.text(0, 0.2, "Surface", fontsize=9, color="#3a7d3a")
    x = np.linspace(2, 9, 100)
    y = -2 + (-3) * (1 - np.exp(-(x - 2) * 1.5))
    ax.plot(x, y, color="#cc2222", lw=2.5)
    # MD arc length annotation
    for i in [10, 30, 50, 70, 90]:
        ax.scatter(x[i], y[i], s=20, color="#cc2222", zorder=5)
    ax.text(3, -1.5, "MD = distance ALONG\nthe wellbore arc", fontsize=10, weight="bold")
    ax.text(3, -2.7, "(1 ft step in this\ndataset)", fontsize=8.5, style="italic", color="#666")
    ax.set_title("Measured Depth (MD)", fontsize=12, color="#cc2222")

    # TVD
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-7, 0.5)
    ax.plot([0, 10], [0, 0], color="#3a7d3a", lw=2.0)
    ax.text(0, 0.2, "Surface", fontsize=9, color="#3a7d3a")
    ax.plot(x, y, color="#cc2222", lw=2.0, alpha=0.5)
    # Vertical drop from a point on the well
    px, py = x[80], y[80]
    ax.plot([px, px], [0, py], color="#1a5f9e", lw=2.5, ls="--")
    ax.annotate("", xy=(px - 0.2, py), xytext=(px - 0.2, 0),
                arrowprops=dict(arrowstyle="<->", color="#1a5f9e", lw=1.2))
    ax.text(px - 0.4, py / 2, "TVD", fontsize=11, color="#1a5f9e",
            weight="bold", va="center", ha="right")
    ax.scatter(px, py, s=50, color="#cc2222", zorder=5)
    ax.text(1.5, -1.5, "TVD = vertical depth\nfrom surface", fontsize=10, weight="bold")
    ax.text(1.5, -2.6, "(Z column is the\nsigned elevation)", fontsize=8.5,
            style="italic", color="#666")
    ax.set_title("True Vertical Depth (TVD)", fontsize=12, color="#1a5f9e")

    # TVT
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-7, 0.5)
    ax.plot([0, 10], [0, 0], color="#3a7d3a", lw=2.0)
    ax.text(0, 0.2, "Surface", fontsize=9, color="#3a7d3a")
    # Geological layers (tilted)
    dip = 0.12
    xs = np.linspace(0, 10, 100)
    layer_ys = [-2.0, -2.6, -3.4, -4.0, -4.5]
    cols = ["#c4d3e6", "#9eb9d6", "#dfb88a", "#c8965d"]
    for i, c in enumerate(cols):
        yu = layer_ys[i] + dip * (xs - 5)
        yl = layer_ys[i + 1] + dip * (xs - 5)
        ax.fill_between(xs, yu, yl, color=c, alpha=0.7)
        ax.plot(xs, yu, color="#888", lw=0.5)
    # Typewell on the left
    tw_x = 1.5
    ax.plot([tw_x, tw_x], [0, -5], color="black", lw=2.5)
    ax.annotate("", xy=(tw_x - 0.15, -5), xytext=(tw_x - 0.15, 0),
                arrowprops=dict(arrowstyle="<->", color="#9333ea", lw=1.2))
    ax.text(tw_x - 0.4, -2.5, "TVT", fontsize=11, color="#9333ea",
            weight="bold", va="center", ha="right", rotation=90)
    # Horizontal well crossing layers
    hwx = np.linspace(3, 9, 100)
    hwy = -3.0 + dip * (hwx - 5) - 0.4
    ax.plot(hwx, hwy, color="#cc2222", lw=2.5)
    # Show TVT of a horizontal point: project to typewell reference
    px, py = hwx[60], hwy[60]
    ax.scatter(px, py, s=60, color="#cc2222", zorder=5)
    ax.plot([px, tw_x], [py, py - dip * (px - 5)], color="#9333ea", lw=1.0, ls=":")
    ax.scatter(tw_x, py - dip * (px - 5), s=40, color="#9333ea", zorder=5)
    ax.text(3.0, -1.3, "TVT = position in the\ngeological column,\nas seen from the typewell",
            fontsize=10, weight="bold")
    ax.set_title("True Vertical Thickness (TVT) — the target", fontsize=12, color="#9333ea")

    save_fig("md_tvd_tvt", fig)


# ---------------------------------------------------------------------------
# Diagram 2b: TVT — typewell vs horizontal well (the definition diagram)
# ---------------------------------------------------------------------------
def draw_tvt_definition():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # Common geology: 5 dipping layers shared by panels (a) and (b)
    layer_y_at_x0 = [-1.2, -2.2, -3.0, -3.8, -4.6, -5.7]
    layer_names = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    layer_colors = ["#c4d3e6", "#9eb9d6", "#7aa1c5",
                    "#dfb88a", "#c8965d", "#a17a55"]
    dip = 0.18
    xs = np.linspace(0, 10, 100)
    horizons_at = lambda y0: y0 + dip * (xs - 5)

    # Each panel has a TVT scale on the LEFT side
    def add_tvt_scale(ax, xpos=0.05):
        """Draw a vertical TVT scale at the chosen x position with ticks at each layer top."""
        ax.plot([xpos, xpos], [-6.2, -0.6], color="#9333ea", lw=2.5)
        # tick marks at each layer top elevation, with TVT label
        tvt_labels = [11200, 11300, 11400, 11500, 11600, 11700]
        for y0, tvt_lab in zip(layer_y_at_x0, tvt_labels):
            ax.plot([xpos - 0.10, xpos + 0.10], [y0, y0],
                    color="#9333ea", lw=1.4)
            ax.text(xpos - 0.18, y0, f"{tvt_lab:,}", color="#9333ea",
                    fontsize=8, ha="right", va="center")
        ax.text(xpos - 0.05, -0.35, "TVT\nscale\n(ft)", color="#9333ea",
                fontsize=9, weight="bold", ha="center", va="bottom")
        ax.annotate("", xy=(xpos, -6.2), xytext=(xpos, -0.6),
                    arrowprops=dict(arrowstyle="-|>", color="#9333ea", lw=1.0))

    def draw_layers(ax):
        layer_lines = [horizons_at(y) for y in layer_y_at_x0]
        layer_lines = [horizons_at(-0.4)] + layer_lines + [horizons_at(-6.2)]
        for i, c in enumerate(["#e8eaed"] + layer_colors):
            ax.fill_between(xs, layer_lines[i], layer_lines[i + 1],
                            color=c, alpha=0.78)
            ax.plot(xs, layer_lines[i + 1], color="#666", lw=0.4)
        # Layer name labels on the right
        for y0, name, c in zip(layer_y_at_x0, layer_names, layer_colors):
            y_rt = y0 + dip * (10 - 5)
            ax.text(10.15, y_rt - 0.18, name, fontsize=7.5, color="#444", va="center")

    # ---------- PANEL (a): TYPEWELL — TVT is direct ----------
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(-0.55, 10.6)
    ax.set_ylim(-7.0, -0.0)
    ax.plot([-0.5, 10], [-0.4, -0.4], color="#3a7d3a", lw=2.0)
    ax.text(-0.5, -0.18, "Surface", fontsize=9, color="#3a7d3a")
    draw_layers(ax)
    add_tvt_scale(ax, xpos=0.0)

    # Typewell at x=2.0 going straight down
    tw_x = 2.0
    ax.plot([tw_x, tw_x], [-0.45, -6.0], color="black", lw=3.2,
            solid_capstyle="round")
    ax.plot([tw_x, tw_x], [-0.45, -6.0], color="white", lw=1.0, ls=(0, (1, 1)))
    ax.text(tw_x, -6.4, "TYPEWELL\n(vertical)",
            ha="center", fontsize=9, color="black", weight="bold")
    # Horizontal connectors from typewell points to TVT scale at that elevation
    for y0 in layer_y_at_x0:
        y_at_tw = horizons_at(y0)[int(tw_x / 10 * (len(xs) - 1))]
        ax.plot([0.0, tw_x], [y_at_tw, y_at_tw],
                color="#9333ea", lw=0.8, ls=":", alpha=0.6)
        ax.scatter(tw_x, y_at_tw, s=20, color="#9333ea", zorder=4)
    ax.text(tw_x + 0.35, -3.3,
            "Typewell drills straight down\nthrough every layer.\n\n"
            "At each row, TVT is\nthe well's physical depth\n(directly measured).",
            fontsize=9, color="#444", va="center")
    ax.set_title("(a) Typewell — TVT is DIRECT", fontsize=12, weight="bold",
                 color="#1a5f9e")

    # ---------- PANEL (b): HORIZONTAL WELL — TVT is INFERRED ----------
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(-0.55, 10.6)
    ax.set_ylim(-7.0, -0.0)
    ax.plot([-0.5, 10], [-0.4, -0.4], color="#3a7d3a", lw=2.0)
    ax.text(-0.5, -0.18, "Surface", fontsize=9, color="#3a7d3a")
    draw_layers(ax)
    add_tvt_scale(ax, xpos=0.0)

    # Horizontal well: build from x=2, curving down to flat lateral that crosses 2 layers
    hw_x = np.linspace(2, 10, 200)
    hw_y = -1.2 + (-2.7) * (1 - np.exp(-(hw_x - 2) * 0.9))
    # Add slight upward drift so well crosses two layers (EGFDU into EGFDL)
    hw_y += 0.10 * (hw_x - 6)
    ax.plot(hw_x, hw_y, color="#cc2222", lw=3.0, solid_capstyle="round")
    ax.text(2.0, -1.05, "HORIZONTAL\nWELL", fontsize=9, color="#cc2222",
            weight="bold")
    # Choose 4 sample points on the lateral and project them to TVT axis
    sample_idx = [60, 110, 160, 195]
    for si in sample_idx:
        px, py = hw_x[si], hw_y[si]
        ax.scatter(px, py, s=55, color="#cc2222", zorder=5,
                   edgecolor="black", linewidth=0.6)
        # Projected TVT = y on the typewell axis would be SAME y value
        # (because in our simple flat-layer schematic, the layer at this physical
        # depth corresponds to the typewell at the same depth)
        # Draw dashed projection back to TVT scale
        ax.plot([px, 0.0], [py, py], color="#9333ea", lw=0.7, ls=":", alpha=0.55)
        ax.scatter(0.0, py, s=22, color="#9333ea", zorder=4)
    ax.text(4.6, -1.6,
            "Bit moves SIDEWAYS through\ndipping layers. At each MD point\n"
            "we ask: which layer is it in?\n→ that layer's TVT in the typewell\n"
            "is the bit's TVT.",
            fontsize=9, color="#444", va="center")
    ax.text(7.5, -5.4, "TVT must be INFERRED\n(from GR matching)",
            fontsize=9, color="#cc2222", weight="bold", style="italic")
    ax.set_title("(b) Horizontal well — TVT is INFERRED", fontsize=12,
                 weight="bold", color="#cc2222")

    # ---------- PANEL (c): Real data — Z vs TVT for the demo well ----------
    ax = axes[2]
    md = demo_hw["MD"].to_numpy()
    z = demo_hw["Z"].to_numpy()
    tvt = demo_hw["TVT"].to_numpy()
    tvt_input = demo_hw["TVT_input"].to_numpy()

    # TWO y-axes on same MD axis: TVT (purple) and Z (blue)
    ax.plot(md, tvt, color="#9333ea", lw=2.0, label="TVT  (geological coord)")
    ax.set_xlabel("MD (ft)")
    ax.set_ylabel("TVT (ft)", color="#9333ea")
    ax.tick_params(axis="y", labelcolor="#9333ea")
    ax.grid(alpha=0.3)
    ax.axvline(demo_ps_md, color="#d05a5a", ls="--", lw=1.0, alpha=0.7,
               label=f"Prediction Start (MD={demo_ps_md:.0f})")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(md, z, color="#1a5f9e", lw=1.5, label="Z  (elevation, geometric)")
    ax2.set_ylabel("Z (ft)  — elevation", color="#1a5f9e")
    ax2.tick_params(axis="y", labelcolor="#1a5f9e")
    ax2.legend(loc="lower right", fontsize=8)

    ax.set_title(f"(c) Real data: {DEMO_WELL}\n"
                 "TVT vs Z along the well — different coordinate systems!",
                 fontsize=11)

    save_fig("tvt_definition", fig)


# ---------------------------------------------------------------------------
# Diagram 3: GR as the geological fingerprint
# ---------------------------------------------------------------------------
def draw_gr_concept():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)

    # (a) schematic: shale vs limestone GR
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # Layers
    layers = [
        (8, 10, "limestone\n(low GR)", "#e8d8a8"),
        (6, 8, "shale\n(HIGH GR)", "#5a4a3a"),
        (4, 6, "sandstone\n(low GR)", "#f0d090"),
        (2, 4, "shale (HIGH GR)", "#5a4a3a"),
        (0, 2, "limestone (low GR)", "#e8d8a8"),
    ]
    for ylo, yhi, name, c in layers:
        ax.add_patch(mpatches.Rectangle((1, ylo), 3, yhi - ylo, fc=c, ec="black", lw=0.5))
        ax.text(1.2, (ylo + yhi) / 2, name, fontsize=8.5, va="center",
                color="white" if "HIGH" in name else "black")
    # GR curve (high in shale, low in others)
    gr_vals = []
    for y in np.linspace(0, 10, 200):
        if 6 < y < 8 or 2 < y < 4:
            gr_vals.append(0.85 + 0.1 * np.random.RandomState(int(y * 10)).standard_normal())
        else:
            gr_vals.append(0.25 + 0.05 * np.random.RandomState(int(y * 10)).standard_normal())
    gr_vals = np.array(gr_vals)
    ys = np.linspace(0, 10, 200)
    ax.plot(5 + 3 * gr_vals, ys, color="#3aa856", lw=1.5)
    ax.text(7.8, 9.5, "GR log\n(API units)", color="#3aa856", fontsize=10, weight="bold")
    ax.set_title("GR is the fingerprint of lithology\n(shale = high GR, limestone/sandstone = low GR)",
                 fontsize=11)

    # (b) real GR curve from demo well's typewell
    ax = axes[1]
    tw = demo_tw.dropna(subset=["GR"])
    ax.plot(tw["GR"], tw["TVT"], color="black", lw=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("GR (API)")
    ax.set_ylabel("TVT (ft)")
    ax.set_title(f"Real example: typewell GR(TVT) for well {DEMO_WELL}\n"
                 "Higher peaks ≈ shale (Eagle Ford!), lower troughs ≈ chalk/limestone",
                 fontsize=11)
    ax.grid(alpha=0.3)
    save_fig("gr_concept", fig)


# ---------------------------------------------------------------------------
# Diagram 4: typewell vs horizontal well GR comparison (real data)
# ---------------------------------------------------------------------------
def draw_typewell_vs_hw():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Typewell GR vs TVT
    ax = axes[0]
    tw = demo_tw.dropna(subset=["GR"])
    ax.plot(tw["GR"], tw["TVT"], color="black", lw=1.0, label="typewell GR")
    ax.invert_yaxis()
    ax.set_xlabel("GR (API)")
    ax.set_ylabel("TVT (ft)")
    ax.set_title(f"{DEMO_WELL} — Typewell\nGR(TVT) — the 'dictionary'")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Horizontal well GR vs MD
    ax = axes[1]
    hw = demo_hw.dropna(subset=["GR"])
    ax.plot(hw["MD"], hw["GR"], color="#3aa856", lw=0.8, label="horizontal-well GR")
    ax.axvline(demo_ps_md, color="#d05a5a", ls="--", lw=1.2, label="Prediction Start (PS)")
    ax.set_xlabel("MD (ft)")
    ax.set_ylabel("GR (API)")
    ax.set_title(f"{DEMO_WELL} — Horizontal well\nGR(MD) — the live observation")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    save_fig("tw_vs_hw", fig)


# ---------------------------------------------------------------------------
# Diagram 5: horizons / formation tops
# ---------------------------------------------------------------------------
def draw_horizons():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # (a) schematic
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    layer_y = [9, 7.5, 6.0, 4.5, 3.0, 1.5]
    layer_names = ["above ANCC", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    layer_colors = ["#e8eaed", "#c4d3e6", "#9eb9d6", "#7aa1c5",
                    "#dfb88a", "#c8965d", "#a17a55"]
    dip = -0.06
    xs = np.linspace(0, 10, 100)
    layer_lines = [10 + dip * (xs - 5)]
    for y in layer_y:
        layer_lines.append(y + dip * (xs - 5))
    layer_lines.append(0 + dip * (xs - 5))
    for i, c in enumerate(layer_colors):
        ax.fill_between(xs, layer_lines[i], layer_lines[i + 1], color=c, alpha=0.85)
        ax.plot(xs, layer_lines[i + 1], color="black", lw=1.0)
    # Label horizons
    for i, name in enumerate(layer_names):
        y = layer_lines[i + 1][-1]
        if i < len(layer_names) - 1:
            ax.annotate("", xy=(8.5, y), xytext=(10, y),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.0))
            ax.text(10.2, y, f"← {name} top", fontsize=8, color="red",
                    va="center", weight="bold")
    ax.text(4.5, 9.5, "Horizons (= formation tops) are the boundaries between layers",
            ha="center", fontsize=10, weight="bold")
    ax.text(4.5, 0.4, "The 6 train-only columns (ANCC..BUDA) give the elevation\n"
                       "of each of these surfaces at every horizontal-well MD point",
            ha="center", fontsize=8.5, color="#444", style="italic")
    ax.set_title("Horizons / Formation Tops", fontsize=12)

    # (b) Real horizon elevations from demo well
    ax = axes[1]
    formations = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
    colors = ["#3b6fbb", "#7aa1c5", "#9eb9d6", "#dfb88a", "#c8965d", "#a17a55"]
    md = demo_hw["MD"].to_numpy()
    for fm, c in zip(formations, colors):
        if fm in demo_hw.columns:
            ax.plot(md, demo_hw[fm], color=c, lw=1.6, label=fm)
    z = demo_hw["Z"].to_numpy()
    ax.plot(md, z, color="black", lw=2.0, label="Wellbore Z")
    ax.axvline(demo_ps_md, color="#d05a5a", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("MD (ft)")
    ax.set_ylabel("Elevation (ft)")
    ax.set_title(f"Real horizons for {DEMO_WELL}\n"
                 "(every row gives each formation's elevation under the bit)")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    save_fig("horizons", fig)


# ---------------------------------------------------------------------------
# Diagram 6: the inverse-problem framing in one cartoon
# ---------------------------------------------------------------------------
def draw_inverse_cartoon():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)

    def box(x, y, w, h, label, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            fc=color, ec="#2b4a7a", lw=1.3,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    # Observed
    box(0.3, 3.5, 2.8, 1.5, "OBSERVED\nGR(MD) along\nhorizontal well", "#fce3d4")
    box(0.3, 1.0, 2.8, 1.5, "REFERENCE\nGR(TVT) from\ntypewell", "#dcecd6")

    # Inverse engine
    box(4.5, 2.5, 3.0, 1.5, "INVERSE\nPROBLEM\n(DTW / Tikhonov\n/ MDN / ...)", "#e2efff")

    # Output
    box(8.5, 2.5, 3.0, 1.5, "RECOVERED\nTVT(MD)\n(the answer)", "#d4e8ff")

    # Arrows
    ax.annotate("", xy=(4.5, 3.7), xytext=(3.1, 4.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(4.5, 3.0), xytext=(3.1, 1.7),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(8.5, 3.3), xytext=(7.5, 3.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
    ax.text(6.0, 5.5, "The ROGII task in one diagram", ha="center",
            fontsize=12, weight="bold")
    save_fig("inverse_cartoon", fig)


draw_master()
draw_md_tvd_tvt()
draw_tvt_definition()
draw_gr_concept()
draw_typewell_vs_hw()
draw_horizons()
draw_inverse_cartoon()


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:100%;max-width:100%'/></figure>"
    )


html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII Glossary — GR / TVT / Typewell / Horizon explained</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    background: #fafbfc; color: #1f2933;
    max-width: 1100px; margin: 0 auto; padding: 28px 36px 80px;
    line-height: 1.62;
  }
  h1 { font-size: 30px; border-bottom: 2px solid #3b6fbb; padding-bottom: 6px; margin-top: 0; }
  h2 { font-size: 22px; color: #2b4a7a; margin-top: 38px;
        border-left: 4px solid #3b6fbb; padding-left: 10px; }
  h3 { font-size: 16px; color: #444; margin-top: 22px; }
  p, li { font-size: 14.5px; }
  .meta { font-size: 12px; color: #6b7280; margin-bottom: 18px; }
  code { background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }
  blockquote { border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
               background: white; color: #2f4d36; font-size: 13.5px; }
  blockquote.warn { border-left-color: #d05a5a; color: #5c2929; background: #fff5f5; }
  blockquote.idea { border-left-color: #b8860b; color: #5c4515; background: #fff9e6; }
  table.data { border-collapse: collapse; width: 100%; font-size: 13px;
               margin: 10px 0 18px; background: white; }
  table.data th, table.data td { border-bottom: 1px solid #e3e7ec;
               padding: 8px 10px; text-align: left; vertical-align: top; }
  table.data th { background: #eef3f9; color: #2b4a7a; }
  figure { margin: 8px 0 24px; }
  hr { border: none; border-top: 1px solid #e3e7ec; margin: 30px 0; }
  a { color: #2b6cb0; }
  .term { display: inline-block; background: #2b4a7a; color: white; padding: 2px 9px;
          border-radius: 5px; font-size: 13px; font-weight: 600; letter-spacing: 0.3px;
          font-family: "SF Mono", "Menlo", monospace; }
  .role-input { color: #cc2222; font-weight: 600; }
  .role-target { color: #9333ea; font-weight: 600; }
  .role-reference { color: #1a5f9e; font-weight: 600; }
  .role-aux { color: #b8860b; font-weight: 600; }
</style>
</head>
<body>

<h1>ROGII Glossary — visually explained</h1>
<div class="meta">
  Every term you need to read the ROGII Wellbore Geology Prediction problem,
  with diagrams. For the formal problem statement see
  <a href="competition_overview.html">competition_overview.html</a>; for the
  data EDA see <a href="eda_report.html">eda_report.html</a>.
</div>

<h2>1. The whole task in one picture</h2>
""" + img("master", "master diagram") + """
<blockquote>
A vertical <b>typewell</b> has already been drilled through the geological column;
it gives us a GR ↔ TVT dictionary. A <b>horizontal well</b> is now being drilled
sideways through the same formation; we measure its GR along the way. <b>The
task</b>: from the horizontal well&apos;s GR observations + the typewell
dictionary + the wellbore trajectory, recover where the bit sits in the
geological column at every measured-depth point. That &ldquo;where&rdquo; is the
<b>TVT</b> coordinate.
</blockquote>

<h2>2. Three depth coordinates: MD, TVD, TVT</h2>
""" + img("md_tvd_tvt", "MD vs TVD vs TVT") + """

<table class="data">
  <thead><tr><th>Term</th><th>What it measures</th><th>Where in the data</th><th>Role</th></tr></thead>
  <tbody>
    <tr>
      <td><span class="term">MD</span></td>
      <td>Distance along the wellbore from surface (arc length). Always increases as we drill further.</td>
      <td>Horizontal CSV column <code>MD</code>. Step is exactly 1 ft per row in every well.</td>
      <td><span class="role-input">input feature</span></td>
    </tr>
    <tr>
      <td><span class="term">Z / TVD</span></td>
      <td>True vertical depth from surface, in geometric (Cartesian) coordinates.</td>
      <td>Horizontal CSV column <code>Z</code> — signed elevation (negative = deeper).</td>
      <td><span class="role-input">input feature</span></td>
    </tr>
    <tr>
      <td><span class="term">TVT</span></td>
      <td>True Vertical Thickness — position in the <i>geological column</i> measured against the typewell&apos;s vertical reference. <b>This is what we predict.</b></td>
      <td>Horizontal CSV column <code>TVT</code> (full in training, hidden post-PS in test). <code>TVT_input</code> is the masked version available at inference.</td>
      <td><span class="role-target">prediction target</span></td>
    </tr>
  </tbody>
</table>

<blockquote class="warn">
<b>TVT ≠ TVD.</b> Two wells in different parts of the field may have the same TVT
(same geological layer) but very different TVD (different surface elevations,
different dipping geometry). TVT lives in the typewell&apos;s frame — it is
designed to be a coordinate that <i>moves with the geology</i>, not with the
geometric depth.
</blockquote>

<h3>2.1 &nbsp;TVT in detail — the same scale, two ways to obtain it</h3>
""" + img("tvt_definition", "TVT typewell vs horizontal well") + """
<p>
This is the single most important concept in the dataset: <b>both wells share
the SAME TVT scale</b> (purple axis on the left of panels a and b) — it is the
&ldquo;floor numbering&rdquo; of the geological building. The only difference
is <i>how</i> we get TVT for each well type.
</p>
<table class="data">
  <thead><tr><th>Well type</th><th>How TVT is obtained</th><th>What it looks like in the CSV</th></tr></thead>
  <tbody>
    <tr>
      <td><span class="term">Typewell (vertical)</span></td>
      <td><b>Direct measurement.</b> The well drills straight down through every
          layer; TVT = its physical depth (panel a). Like an elevator that
          records its floor as it descends.</td>
      <td><code>TVT</code> column, sampled every 0.5 ft of true depth. Range
          ~11,224 – 11,871 ft for the demo well.</td>
    </tr>
    <tr>
      <td><span class="term">Horizontal well (lateral)</span></td>
      <td><b>Must be inferred.</b> The bit travels sideways through dipping
          layers (panel b). At each MD point we ask: &ldquo;which layer is the
          bit in?&rdquo; That layer&apos;s TVT in the typewell <b>is</b> the
          bit&apos;s TVT — but to identify the layer we need to match GR
          signatures.</td>
      <td><code>TVT</code> column (training only); <code>TVT_input</code>
          available only before PS at inference. This is the column we predict.</td>
    </tr>
  </tbody>
</table>
<p>
Panel (c) shows real data from the demo well: <code>TVT</code> (purple) lives
in the geological coordinate frame and stays within a relatively narrow band
(the Eagle Ford zone, ~11,200&ndash;11,800 ft); <code>Z</code> (blue) is the
geometric elevation in the surveying coordinate frame and tracks the
wellbore&apos;s actual physical position. They are clearly <b>two different
coordinates</b> of the same MD axis. The competition asks us to recover the
purple curve given the geometric Z, the observed GR, and the typewell
GR(TVT) dictionary.
</p>

<h2>3. GR — the gamma-ray log, geology&apos;s fingerprint</h2>
""" + img("gr_concept", "GR concept") + """
<p>
GR is the natural gamma radiation emitted by the rock the sensor passes through.
Different lithologies have different characteristic GR levels:
</p>
<ul>
  <li><b>Shales</b> — high GR (organic-rich; contain potassium, thorium, uranium).
      Eagle Ford in this dataset is a shale → high GR.</li>
  <li><b>Limestones / chalks / sandstones</b> — low GR.</li>
</ul>
<p>
Because every formation has its own GR signature, the GR <i>sequence</i> with
depth is a unique fingerprint of the local stratigraphy. That fingerprint is
what we match between horizontal and vertical wells.
</p>

<h2>4. Typewell vs horizontal well</h2>
""" + img("tw_vs_hw", "Typewell vs horizontal-well GR") + """
<table class="data">
  <thead><tr><th>Term</th><th>What it is</th><th>Its role here</th></tr></thead>
  <tbody>
    <tr>
      <td><span class="term">Typewell</span></td>
      <td>A nearby <b>vertical</b> well drilled through the full geological column. Sampled every 0.5 ft of TVT.</td>
      <td><span class="role-reference">reference / dictionary</span> — provides GR(TVT) so we can look up the TVT of a given GR signature.</td>
    </tr>
    <tr>
      <td><span class="term">Horizontal well</span></td>
      <td>The <b>lateral</b> well we are predicting. Sampled every 1 ft of MD.</td>
      <td><span class="role-input">the well we predict on</span> — gives the live GR observation + trajectory.</td>
    </tr>
  </tbody>
</table>

<p>
The two GR plots look very different on purpose: the typewell&apos;s y-axis is
TVT (depth, inverted), the horizontal well&apos;s x-axis is MD (along-hole
distance). The competition task is to figure out, for every point on the
right, which point on the left it corresponds to.
</p>

<h2>5. Horizons / formation tops</h2>
""" + img("horizons", "horizons") + """
<p>
A <b>horizon</b> is a geological boundary — the surface that separates two
formations. In this dataset, the six formations from shallow to deep are:
</p>
<ol>
  <li><span class="term">ANCC</span> — Anacacho (limestone, top of the column)</li>
  <li><span class="term">ASTNU</span> — Austin Chalk (Upper)</li>
  <li><span class="term">ASTNL</span> — Austin Chalk (Lower)</li>
  <li><span class="term">EGFDU</span> — Eagle Ford (Upper) — <i>the productive shale zone</i></li>
  <li><span class="term">EGFDL</span> — Eagle Ford (Lower) — <i>still the productive zone</i></li>
  <li><span class="term">BUDA</span> — Buda Limestone (base of interest)</li>
</ol>

<p>
In the <b>training</b> horizontal CSVs, each row contains six extra columns
(one per formation) giving the absolute elevation of each formation top at the
trajectory point. These describe the geological structure under the bit and
are <b>removed in the test set</b> — they would leak the answer.
</p>

<blockquote class="idea">
<b>Use the formation-top columns as auxiliary supervision.</b> They are train-only,
so they can&apos;t be inference inputs — but an ML model trained to predict
TVT <i>and</i> the six formation-top elevations simultaneously learns better
representations than one trained on TVT alone.
</blockquote>

<h2>6. Prediction Start (PS)</h2>
<p>
<span class="term">PS</span> is the MD index where the known <code>TVT_input</code>
stops and the prediction zone begins. In the master diagram above, the dashed
red line marks PS. Before PS, the model can see the true TVT (via
<code>TVT_input</code>); after PS, it must predict.
</p>
<p>
Median in this dataset: PS sits around 25% of the way through the well, so
the prediction zone is about 75% of the lateral. See
<a href="eda_report.html">eda_report.html</a> §4 for the exact distribution.
</p>

<h2>7. The whole thing as an inverse problem</h2>
""" + img("inverse_cartoon", "inverse problem cartoon") + """
<p>
Given an observed GR sequence along the horizontal well and a typewell
GR-vs-TVT dictionary, we want to recover the TVT(MD) function. This is
literally <b>inversion</b>: solve for the unknown geological position given
forward-model observations. Concrete methods (DTW alignment, Tikhonov-regularized
Gauss-Newton, sequential Kalman/particle filters, MDN multi-modal inversion)
are graded in
<a href="competition_overview.html">competition_overview.html</a> §8.
</p>

<h2>8. Cheat-sheet table — every term&apos;s role</h2>
<table class="data">
  <thead><tr><th>Term</th><th>Symbol / column</th><th>Role</th><th>Available at inference?</th></tr></thead>
  <tbody>
    <tr><td>Measured Depth</td><td><code>MD</code></td>
        <td><span class="role-input">input</span> (sequence index)</td>
        <td>Yes</td></tr>
    <tr><td>Trajectory coordinates</td><td><code>X, Y, Z</code></td>
        <td><span class="role-input">input</span> (geometry)</td>
        <td>Yes</td></tr>
    <tr><td>Gamma Ray</td><td><code>GR</code></td>
        <td><span class="role-input">input</span> (the observation)</td>
        <td>Yes (~28% missing on average)</td></tr>
    <tr><td>Known TVT before PS</td><td><code>TVT_input</code></td>
        <td><span class="role-input">input</span> (warm start)</td>
        <td>Yes (before PS only)</td></tr>
    <tr><td>True Vertical Thickness</td><td><code>TVT</code></td>
        <td><span class="role-target">target</span></td>
        <td>No (this is what we predict)</td></tr>
    <tr><td>Typewell GR</td><td>typewell <code>GR</code></td>
        <td><span class="role-reference">reference</span> (vertical dictionary)</td>
        <td>Yes</td></tr>
    <tr><td>Typewell TVT</td><td>typewell <code>TVT</code></td>
        <td><span class="role-reference">reference</span> (depth axis for the dictionary)</td>
        <td>Yes</td></tr>
    <tr><td>Typewell geology label</td><td>typewell <code>Geology</code></td>
        <td><span class="role-reference">reference</span> (sparse formation labels)</td>
        <td>Yes</td></tr>
    <tr><td>Formation tops (6)</td><td><code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code></td>
        <td><span class="role-aux">auxiliary training supervision</span></td>
        <td><b>No</b> — train only</td></tr>
  </tbody>
</table>

<hr/>
<p class="meta">
Companion reports: <a href="eda_report.html">eda_report.html</a> (full EDA) ·
<a href="competition_overview.html">competition_overview.html</a>
(formal inverse-problem framing + methods) ·
<a href="geological_map.html">geological_map.html</a> (spatial geology) ·
<a href="wells_on_geology.html">wells_on_geology.html</a> (wells overlaid on
geology) · <a href="index.html">index.html</a> (knowledge-base front page).
</p>

</body>
</html>
"""

out = REPORT_DIR / "glossary.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
