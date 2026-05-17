"""Generate report/competition_overview.html — an English explainer of the ROGII
Wellbore Geology Prediction task, framed as a geophysical inverse problem.

The report integrates:
  * Formal inverse-problem formulation (Tikhonov / Gauss-Newton).
  * Concrete domain insights drawn from competition discussion threads
    (697431, 699853, 699326) and the LWD_inversion reference repo
    (https://github.com/hhschumann/LWD_inversion).
  * A graded method portfolio: from constant extrapolation to ML+inversion
    hybrids.

Diagrams are matplotlib-generated and embedded as base64 PNGs so the file is
self-contained.

Equations render via MathJax (CDN). Raw TeX is still readable offline.
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

REPORT_DIR = Path("/home/tom99763/ROGII/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGS: dict[str, str] = {}


def save_fig(name: str, fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Diagram 1: forward vs inverse
# ---------------------------------------------------------------------------
def draw_forward_inverse():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")

    def box(x, y, w, h, label, color, ec="#2b4a7a"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec=ec, lw=1.2
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, weight="bold")

    def arrow(x1, y1, x2, y2, label, color="#1f2933"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.07, label, ha="center", fontsize=9, color=color)

    # Top row: forward
    box(0.02, 0.62, 0.22, 0.28, "Formation\nmodel  m", "#e2efff")
    box(0.40, 0.62, 0.22, 0.28, "Forward operator\nF(m)", "#dcecd6")
    box(0.78, 0.62, 0.20, 0.28, "Predicted\nGR(MD)", "#ffe6c8")
    arrow(0.24, 0.76, 0.40, 0.76, "physics")
    arrow(0.62, 0.76, 0.78, 0.76, "")

    # Bottom row: inverse
    box(0.02, 0.10, 0.22, 0.28, "Observed\nGR(MD)", "#ffe6c8")
    box(0.40, 0.10, 0.22, 0.28, "Inverse operator\n$F^{-1}$ (regularized)", "#fce3d4")
    box(0.78, 0.10, 0.20, 0.28, "Recovered\nTVT(MD)", "#e2efff")
    arrow(0.24, 0.24, 0.40, 0.24, "minimize misfit")
    arrow(0.62, 0.24, 0.78, 0.24, "")

    ax.text(0.50, 0.55, "↑ what physics gives us  ·  ↓ what the competition asks for ↓",
            ha="center", fontsize=10, color="#6b7280", style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("forward_inverse", fig)


# ---------------------------------------------------------------------------
# Diagram 2: shift+stretch matching (mirrors user's image #2)
# ---------------------------------------------------------------------------
def draw_shift_stretch():
    rng = np.random.default_rng(7)
    n = 200
    z = np.linspace(0, 10, n)
    base = 1.0 * np.sin(z) + 0.7 * np.sin(2.7 * z + 0.3) + 0.3 * np.sin(5 * z)
    base = base + 0.15 * rng.standard_normal(n)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(base, z, color="#3b6fbb", lw=1.1)
    ax.invert_yaxis()
    ax.set_title("Vertical reference\n(typewell) GR vs TVT")
    ax.set_xlabel("GR")
    ax.set_ylabel("TVT (depth)")
    ax.set_xticks([])

    ax = axes[1]
    snippet = base[80:110] + 0.05 * rng.standard_normal(30)
    md = np.arange(len(snippet))
    ax.plot(md, snippet, color="#d05a5a", lw=1.6)
    ax.set_title("Horizontal-well GR snippet\n‘where am I?’")
    ax.set_xlabel("MD (within snippet)")
    ax.set_ylabel("GR")

    ax = axes[2]
    ax.plot(base, z, color="#3b6fbb", lw=1.1, alpha=0.5)
    # Snippet shifted to match TVT range 4..5.5 → indices 80..110 in z
    snip_z = z[80:110]
    ax.plot(snippet, snip_z, color="#d05a5a", lw=1.8)
    ax.invert_yaxis()
    ax.set_title("Best match → ‘you are here’\n(snippet aligned on TVT axis)")
    ax.set_xlabel("GR")
    ax.set_ylabel("TVT")
    ax.set_xticks([])
    ax.axhspan(z[80], z[110], color="#d05a5a", alpha=0.08)
    ax.annotate(
        "matched TVT range",
        xy=((base.max() + base.min()) / 2, (z[80] + z[110]) / 2),
        xytext=((base.max() + base.min()) / 2 + 2.5, (z[80] + z[110]) / 2 - 1.5),
        arrowprops=dict(arrowstyle="->", color="#d05a5a"),
        color="#d05a5a", fontsize=9,
    )
    save_fig("shift_stretch", fig)


# ---------------------------------------------------------------------------
# Diagram 3: typewells of nearby wells are offset copies (mirrors user's image #3)
# ---------------------------------------------------------------------------
def draw_offset_typewells():
    rng = np.random.default_rng(11)
    n = 400
    base = (
        1.0 * np.sin(np.linspace(0, 18, n))
        + 0.4 * np.sin(np.linspace(0, 50, n))
        + 0.25 * rng.standard_normal(n)
    )
    base = 100 + 25 * base

    fig, ax = plt.subplots(figsize=(11, 4.5))
    offsets = [0, 50, 95]
    colors = ["#3b6fbb", "#e9924b", "#5cb874"]
    labels = ["typewell A (well 5fffa282)", "typewell B (well e0f36d98)", "typewell C (well 3a86fe8d)"]
    for off, c, lbl in zip(offsets, colors, labels):
        ax.plot(np.arange(n) + off, base, color=c, lw=1.1, label=lbl)
    ax.set_xlabel("relative depth index (MD-equivalent)")
    ax.set_ylabel("GR")
    ax.set_title("Nearby wells' typewells are the SAME underlying log — only depth-offset copies")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Highlight a matching feature
    feat = 220
    for off, c in zip(offsets, colors):
        ax.axvspan(feat + off - 4, feat + off + 4, color=c, alpha=0.12)
    ax.text(
        feat + offsets[1], base.max() + 4,
        "same geological marker → appears at shifted positions in each well",
        ha="center", fontsize=9, color="#6b7280", style="italic",
    )
    save_fig("offset_typewells", fig)


# ---------------------------------------------------------------------------
# Diagram 4: sequential inversion loop (mirrors user's image #4)
# ---------------------------------------------------------------------------
def draw_inversion_loop():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#2b4a7a", lw=1.2
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, color="#1f2933"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
        )

    box(0.02, 0.78, 0.24, 0.16, "Offset / pilot well logs", "#e2efff")
    box(0.36, 0.78, 0.28, 0.16, "Input LWD data\n(GR at MD t)", "#fce3d4")
    box(0.72, 0.78, 0.24, 0.16, "Initial guess m₀\nfrom previous point", "#fff8d4")

    box(0.36, 0.55, 0.28, 0.14, "Build inverse model", "#dcecd6")
    arrow(0.14, 0.78, 0.36, 0.69)
    arrow(0.50, 0.78, 0.50, 0.69)
    arrow(0.84, 0.78, 0.64, 0.69)

    box(0.36, 0.36, 0.28, 0.14, "Gauss–Newton step\n(eq. 5–6)", "#dcecd6")
    arrow(0.50, 0.55, 0.50, 0.50)

    box(0.36, 0.18, 0.28, 0.14, "Compare F(m) vs d_obs", "#dcecd6")
    arrow(0.50, 0.36, 0.50, 0.32)

    box(0.04, 0.18, 0.22, 0.14, "Adjust model →\nrepeat", "#fff8d4")
    arrow(0.36, 0.25, 0.26, 0.25)
    arrow(0.15, 0.32, 0.36, 0.45)

    box(0.72, 0.18, 0.24, 0.14, "Output TVT at MD t →\nseed for MD t+1", "#fce3d4")
    arrow(0.64, 0.25, 0.72, 0.25)
    # Feedback arrow upward
    ax.annotate(
        "", xy=(0.84, 0.78), xytext=(0.84, 0.32),
        arrowprops=dict(arrowstyle="->", color="#3aa856", lw=1.6, ls="dashed"),
    )
    ax.text(0.86, 0.55, "next MD\npoint", color="#3aa856", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("inversion_loop", fig)


# ---------------------------------------------------------------------------
# Diagram 5: method spectrum
# ---------------------------------------------------------------------------
def draw_method_spectrum():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    methods = [
        ("Constant\nextrapolation", "#fff3e0", 0),
        ("Linear in Z\n(local slope)", "#ffe0b8", 1),
        ("DTW / cross-corr\nGR ↔ typewell", "#d8e8d4", 2),
        ("Tikhonov\ninversion (GN)", "#a8d5a0", 3),
        ("State-space:\nKalman / particle", "#7cc070", 4),
        ("ML sequence\n(1D-CNN / TF)", "#52ad58", 5),
        ("Hybrid: ML\n+ inversion", "#2a8847", 6),
    ]
    for label, color, i in methods:
        x = 0.02 + i * 0.14
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.30), 0.12, 0.40, boxstyle="round,pad=0.02",
            fc=color, ec="#2b4a7a", lw=1.0,
        ))
        ax.text(x + 0.06, 0.50, label, ha="center", va="center", fontsize=8.5)
    ax.annotate(
        "", xy=(0.99, 0.18), xytext=(0.01, 0.18),
        arrowprops=dict(arrowstyle="->", color="#1f2933", lw=1.4),
    )
    ax.text(0.01, 0.08, "trivial baseline", fontsize=9, color="#6b7280")
    ax.text(0.99, 0.08, "competitive solution", fontsize=9, ha="right", color="#6b7280")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig("method_spectrum", fig)


# ---------------------------------------------------------------------------
# Diagram 6: typewell stacking (idea — combine multiple nearby typewells)
# ---------------------------------------------------------------------------
def draw_typewell_stack():
    rng = np.random.default_rng(3)
    z = np.linspace(0, 1, 400)
    base = (
        np.sin(2 * np.pi * 3 * z)
        + 0.5 * np.sin(2 * np.pi * 9 * z)
    )
    base = 90 + 30 * base
    noisy_versions = []
    shifts = [0, -0.012, 0.008, -0.005, 0.014]
    for s in shifts:
        zs = z + s
        # interpolate base at zs
        v = np.interp(zs, z, base) + 4 * rng.standard_normal(len(z))
        noisy_versions.append(v)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    for i, v in enumerate(noisy_versions):
        ax.plot(z, v, alpha=0.55, lw=0.9, label=f"typewell {i + 1}")
    ax.set_title("Five offset-copy typewells (raw)")
    ax.set_xlabel("normalized depth")
    ax.set_ylabel("GR")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    stack = np.median(np.array(noisy_versions), axis=0)
    ax.plot(z, base, color="#3b6fbb", lw=1.0, alpha=0.6, label="true reference")
    ax.plot(z, stack, color="#d05a5a", lw=1.4, label="stacked median (5 typewells)")
    ax.set_title("Stack neighbour typewells → cleaner GR reference")
    ax.set_xlabel("normalized depth")
    ax.set_ylabel("GR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    save_fig("typewell_stack", fig)


draw_forward_inverse()
draw_shift_stretch()
draw_offset_typewells()
draw_inversion_loop()
draw_method_spectrum()
draw_typewell_stack()


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def img(name: str, alt: str = "", width: str = "100%") -> str:
    return (
        f"<figure><img src='data:image/png;base64,{FIGS[name]}' alt='{alt}' "
        f"style='width:{width};max-width:100%'/></figure>"
    )


html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ROGII Wellbore Geology Prediction — Problem &amp; Methods</title>
<script>
  MathJax = {
    tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]}
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
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
  h3 { font-size: 16px; color: #444; margin-top: 24px; }
  p, li { font-size: 14.5px; }
  .meta { font-size: 12px; color: #6b7280; margin-bottom: 18px; }
  code { background: #eef0f2; padding: 1px 5px; border-radius: 3px; font-size: 13px; }
  pre { background: #1f2933; color: #f3f4f6; padding: 12px 14px; border-radius: 6px;
        overflow-x: auto; font-size: 12.5px; }
  blockquote { border-left: 3px solid #5cb874; margin: 14px 0; padding: 10px 16px;
               background: white; color: #2f4d36; font-size: 13.5px; }
  blockquote.warn { border-left-color: #d05a5a; color: #5c2929; background: #fff5f5; }
  blockquote.idea { border-left-color: #b8860b; color: #5c4515; background: #fff9e6; }
  figure { margin: 14px 0 24px; }
  table.data { border-collapse: collapse; width: 100%; font-size: 12.5px;
               margin: 10px 0 18px; background: white; }
  table.data th, table.data td { border-bottom: 1px solid #e3e7ec;
               padding: 7px 10px; text-align: left; vertical-align: top; }
  table.data th { background: #eef3f9; color: #2b4a7a; }
  .pill { display: inline-block; background: #e2efff; color: #2b4a7a; padding: 1px 8px;
          border-radius: 8px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
          text-transform: uppercase; }
  .pill.warn { background: #ffe2e2; color: #8c2929; }
  .pill.idea { background: #fff5cc; color: #6b4f00; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 12px; margin: 18px 0; }
  .kpi { background: white; border: 1px solid #e3e7ec; border-radius: 8px; padding: 12px 14px; }
  .kpi .v { font-size: 18px; font-weight: 600; color: #1f2933; }
  .kpi .l { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
  hr { border: none; border-top: 1px solid #e3e7ec; margin: 26px 0; }
  a { color: #2b6cb0; }
</style>
</head>
<body>

<h1>ROGII Wellbore Geology Prediction — Problem &amp; Methods</h1>
<div class="meta">
  Competition: <a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction">
  rogii-wellbore-geology-prediction</a>
  &middot; This document complements <code>report/eda_report.html</code> (data EDA) with a
  problem framing and method portfolio.
</div>

<h2>1. TL;DR</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="l">Real task</div><div class="v">Geophysical inversion</div></div>
  <div class="kpi"><div class="l">Model variable</div><div class="v">TVT(MD) sequence</div></div>
  <div class="kpi"><div class="l">Observation</div><div class="v">GR(MD) + typewell GR(TVT)</div></div>
  <div class="kpi"><div class="l">Loss / metric</div><div class="v">RMSE on post-PS TVT</div></div>
</div>

<blockquote>
The ROGII competition <i>looks</i> like a tabular regression task — &ldquo;predict <code>TVT</code> from
features.&rdquo; It is more usefully framed as a <b>geophysical inverse problem</b>: a horizontal well has
drilled through a (partially known) geological structure; we observe its gamma-ray (GR) signature along
measured depth (MD); we have a vertical reference GR–depth log (the <i>typewell</i>); we want to recover
where along the geological column the well actually is at each MD point — i.e.&nbsp;the per-point
<code>TVT</code>. This is the same inverse problem solved (manually or with software like ROGII&apos;s
StarSteer) every day on a drilling rig.
</blockquote>

<h2>2. The Physical Setup</h2>
<p>A horizontal well is drilled out of a vertical &ldquo;pilot&rdquo; or &ldquo;type&rdquo; well that
penetrates the geological column at one known location. As the bit travels along the lateral, an
LWD (Logging While Drilling) gamma-ray tool records GR at every measured-depth (MD) step. Because the
rock&apos;s GR signature varies vertically (each formation has a characteristic GR pattern), the GR the
bit sees tells us which formation it is currently in — and therefore where it is in <i>true vertical
thickness</i> (TVT).</p>

<p><b>Why this is hard.</b> The wellbore is mostly horizontal but the geology is dipping at unknown angles;
small dips translate to large TVT changes over a 1&ndash;2&nbsp;km lateral. The GR observed at the bit is
noisy. Some segments of GR are missing. And the assigned typewell only covers a finite TVT range.</p>

""" + img("forward_inverse", "forward vs inverse") + """

<h2>3. Dataset in One Diagram</h2>
<table class="data">
  <thead><tr><th>Symbol</th><th>Meaning</th><th>Where</th></tr></thead>
  <tbody>
    <tr><td>$d_{obs} = \\mathrm{GR}(\\mathrm{MD})$</td>
        <td>Observed gamma ray along the horizontal well</td>
        <td>Horizontal CSV — column <code>GR</code> (~29% missing on average)</td></tr>
    <tr><td>$\\mathrm{GR}_v(\\mathrm{TVT})$</td>
        <td>Vertical reference GR vs depth</td>
        <td>Typewell CSV — columns <code>GR, TVT</code></td></tr>
    <tr><td>$\\mathbf{x}(\\mathrm{MD}) = (X, Y, Z)$</td>
        <td>Wellbore trajectory at each MD point</td>
        <td>Horizontal CSV — columns <code>X, Y, Z</code></td></tr>
    <tr><td>$h(\\mathrm{MD}) = \\mathrm{TVT}$</td>
        <td><b>The model we want to recover</b></td>
        <td>Horizontal CSV — column <code>TVT</code> (only present in train)</td></tr>
    <tr><td>$h(\\mathrm{MD}<\\mathrm{PS})$</td>
        <td>TVT known up to the Prediction Start point</td>
        <td>Horizontal CSV — column <code>TVT_input</code> (NaN after PS)</td></tr>
    <tr><td>$\\partial_{MD} h \\approx 0$ <i>(prior)</i></td>
        <td>Geology is mostly flat at the per-foot scale</td>
        <td>From EDA: median |dTVT/dMD| ≈ 0.06; long tail of steep dips</td></tr>
  </tbody>
</table>

<h2>4. The Geosteering Intuition: Shift &amp; Stretch</h2>
""" + img("shift_stretch", "shift and stretch matching") + """

<p>Take a short window of horizontal-well GR (red). It has to come from somewhere on the vertical
reference (blue). The geological &ldquo;where am I?&rdquo; question is answered by finding the depth
interval on the typewell that best matches the snippet&apos;s GR pattern, allowing for a small affine
warp (shift + stretch) to account for slight dip. The matched TVT range is your wellbore&apos;s current
TVT.</p>

<p>This is exactly what a geosteering engineer does by eye, slide by slide, on the rig. Algorithmically
it is <b>subsequence matching with affine alignment</b> — a problem with mature tooling (DTW, sliding
correlation, FFT-based phase correlation).</p>

<h2>5. Formal Inverse-Problem Formulation</h2>
<p>Let <em>m</em> = $h(\\mathrm{MD})$ be the unknown TVT sequence we want to recover. Let
$d_{obs}$ be the observed horizontal GR vector. Define the forward operator</p>

\\[
F(m) := \\mathrm{GR}_v\\!\\bigl(h(\\mathrm{MD})\\bigr)
\\]

<p>i.e. &ldquo;the GR you would expect to see at the bit if it were drilling at the proposed TVT, looked
up from the typewell.&rdquo; If $F(m)$ matched $d_{obs}$ for every MD point, we would be done — but
$F$ is approximate (the typewell is noisy and only an analogue of the local geology), so we solve a
regularized optimization instead:</p>

\\[
\\boxed{\\min_h \\; \\Phi(h) \\;=\\; \\underbrace{\\|W_d(F(h)-d_{obs})\\|_2^2}_{\\text{data misfit }\\Phi_d}
\\;+\\; \\beta \\underbrace{\\|W_m h\\|_2^2}_{\\text{smoothness prior }\\Phi_m}}
\\]

<p>$W_d$ is a data-weight matrix (e.g.&nbsp;inverse-variance of GR noise, zero-weight rows for missing
GR). $W_m$ encodes a smoothness prior on the TVT path — typically a finite-difference operator so
$\\|W_m h\\|^2$ penalizes curvature. $\\beta$ trades fidelity against smoothness and is picked via the
L-curve criterion or cross-validation.</p>

<h3>Gauss&ndash;Newton iteration</h3>
<p>Linearize $F$ around the current iterate via its Jacobian $J = \\partial F / \\partial h$. The gradients
of the data-misfit and prior terms are:</p>

\\[
\\nabla \\Phi_d = J^\\top W_d^\\top W_d (F(h)-d_{obs}), \\qquad
\\nabla \\Phi_m = W_m^\\top W_m h, \\qquad
\\nabla \\Phi = \\nabla \\Phi_d + \\beta \\nabla \\Phi_m \\equiv g.
\\]

<p>The Gauss&ndash;Newton Hessian and the per-iteration model update are:</p>

\\[
H_{GN} = J^\\top W_d^\\top W_d J + \\beta W_m^\\top W_m, \\qquad
H_{GN}\\, \\Delta h = -g.
\\]

<p>This is exactly the formulation used in the open-source LWD inversion reference repo by Schumann
(2021): <a href="https://github.com/hhschumann/LWD_inversion">github.com/hhschumann/LWD_inversion</a>.
A faithful 1-D implementation is &lt; 200 lines of NumPy.</p>

<blockquote class="warn">
<b>Known weakness.</b> Linearized Gauss&ndash;Newton is prone to local minima — if the initial guess sits
near a different geological layer than the truth, the iteration locks in there. Mitigation: <b>start
from the Prediction-Start TVT and march one MD point at a time, using the previous point&apos;s
solution as the initial guess</b> (next section).
</blockquote>

<h2>6. Sequential Inversion (Marching Along MD)</h2>
""" + img("inversion_loop", "inversion loop") + """

<p>The standard LWD-EM workflow marches the inversion forward one MD step at a time, reusing the
previous step&apos;s recovered TVT as the initialization for the next. This both regularizes the problem
(a smoothness prior baked into the iteration order) and dramatically speeds it up (each step needs
only a few Gauss&ndash;Newton iterations).</p>

<p>In state-space language, this is a sequential Bayesian filter on the latent TVT process:</p>

\\[
\\underbrace{h_{t} \\,=\\, h_{t-1} + v_t}_{\\text{drift = local dip}} ,\\qquad
\\underbrace{\\mathrm{GR}_t \\,=\\, \\mathrm{GR}_v(h_t) + \\epsilon_t}_{\\text{forward model}}
\\]

<p>Concrete algorithms that fit this template:</p>
<ul>
  <li><b>Extended Kalman filter</b> — linearize $\\mathrm{GR}_v(h)$ around the predicted mean. Cheap and
      online, but suffers when the GR slope is steep (high $|d\\mathrm{GR}_v / dh|$).</li>
  <li><b>Unscented Kalman filter</b> — better for the highly nonlinear $\\mathrm{GR}_v(h)$ lookup.</li>
  <li><b>Particle filter</b> — N hypotheses propagated forward, reweighted by GR likelihood. Robust to
      multi-modal posteriors (which happen often when GR is locally ambiguous).</li>
</ul>

<h2>7. Two Critical Dataset-Specific Insights</h2>

<h3>7.1 &nbsp;Nearby wells&apos; typewells are offset copies of the same log</h3>
""" + img("offset_typewells", "offset typewells") + """
<p>In the survey area, neighbouring wells were drilled into the same dipping geological column. Their
assigned typewells therefore record the <i>same underlying GR pattern</i>, just shifted in depth (and
mildly stretched by local dip). From the EDA: the 773 training typewells share only 28 distinct geology
labels, dominated by the same 6 formations everywhere.</p>

<p><b>Practical consequence:</b> for any given horizontal well, the assigned typewell is only one of many
admissible vertical references. <i>Stacking the nearest K typewells</i> (after a simple cross-correlation
alignment) yields a much cleaner reference signal than any single typewell:</p>

""" + img("typewell_stack", "typewell stacking") + """

<blockquote class="idea">
<b>Idea.</b> Pre-compute a &ldquo;super-typewell&rdquo; for each well by aligning + median-stacking the K
nearest typewells (K ≈ 5&ndash;10 by XY distance). Use it everywhere the assigned typewell appears in
your pipeline. This single change tends to bump inversion-based methods materially.
</blockquote>

<h3>7.2 &nbsp;The 6 formation-top columns are training-only ground truth — perfect auxiliary signal</h3>
<p>In the training horizontal CSVs the columns <code>ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA</code> hold the
absolute elevation of each formation top at the current trajectory point. They encode the true
geological structure that the model must recover; they are missing in test. Two ways to exploit them:</p>
<ol>
  <li><b>Auxiliary multi-task targets</b> for any ML model — predict the 6 elevations as extra heads.
      The internal representations learned have direct geological meaning and transfer to TVT.</li>
  <li><b>Forward-model bootstrap</b>: at training time, you know exactly which formation the bit is in
      at each MD point. Use that to teach a forward model $F$ (formation, position $\\to$ GR) that
      generalizes beyond the single typewell lookup.</li>
</ol>

<h2>8. Method Portfolio (Graded by Ambition)</h2>
""" + img("method_spectrum", "method spectrum") + """

<table class="data">
  <thead><tr><th>Method</th><th>What it does</th><th>Pros</th><th>Cons</th></tr></thead>
  <tbody>
    <tr><td><span class="pill">Baseline</span><br>Constant extrapolation</td>
        <td>Predict last known TVT for all post-PS rows.</td>
        <td>Trivial. Sets the metric floor.</td>
        <td>Median RMSE ≈ 13&nbsp;ft (see <code>baseline_rmse.csv</code>).</td></tr>
    <tr><td><span class="pill">Baseline</span><br>Linear in Z</td>
        <td>Fit TVT &asymp; a + b·Z on last 50 known rows, extrapolate.</td>
        <td>Cuts median RMSE roughly in half versus constant.</td>
        <td>Fails when the lateral changes dip direction.</td></tr>
    <tr><td><span class="pill">Classical</span><br>DTW / cross-correlation</td>
        <td>Subsequence-match horizontal GR to typewell GR (or super-typewell).</td>
        <td>Domain-faithful; no training needed; interpretable.</td>
        <td>Sensitive to GR noise; needs window-size tuning; offset-typewell stacking helps a lot.</td></tr>
    <tr><td><span class="pill">Classical</span><br>Tikhonov / Gauss&ndash;Newton inversion</td>
        <td>Solve eq. (1)&ndash;(6) above with $W_m$ = 2nd-difference smoothness operator.</td>
        <td>Mathematically clean; supports missing GR via row-zeroed $W_d$.</td>
        <td>Local-minimum trap; needs good init (use DTW or previous-point solution).</td></tr>
    <tr><td><span class="pill">Probabilistic</span><br>Particle / Kalman filter</td>
        <td>Sequential state-space inference on TVT with GR likelihood.</td>
        <td>Handles multimodal posteriors, missing data, online.</td>
        <td>Particle weights collapse if the forward model is too peaky.</td></tr>
    <tr><td><span class="pill">ML</span><br>1D-CNN / Bi-LSTM / Transformer</td>
        <td>Sequence-to-sequence regression of TVT from $(\\Delta XYZ, \\mathrm{GR}, \\mathrm{GR}_{mask}, \\mathrm{TVT}_{input}, \\mathrm{GR}_v(\\cdot))$.</td>
        <td>End-to-end; can absorb soft priors via auxiliary heads (formation tops, geology label, dip).</td>
        <td>Needs careful per-well CV; 773 wells is &ldquo;small&rdquo; for deep models without strong priors.</td></tr>
    <tr><td><span class="pill idea">Hybrid</span><br>Inversion seed → ML refinement</td>
        <td>Run classical inversion to produce a TVT trajectory + uncertainty, then learn a residual.</td>
        <td>Combines physical fidelity with ML&apos;s pattern flexibility; the residual is small and
            well-behaved.</td>
        <td>Two-stage pipeline; need to manage error propagation.</td></tr>
  </tbody>
</table>

<h2>9. Discussion Threads That Frame This View</h2>
<p>The inverse-problem framing came together via three Kaggle discussion threads on the competition
forum (and the LWD_inversion reference repo). Direct links:</p>
<ul>
  <li><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431">discussion/697431</a>
      — visual explanation of GR projection from MD onto TVT (3D MD–TVT–GR plots; horizontal GR
      color-coded by MD must align with typewell GR on the TVT axis).</li>
  <li><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853">discussion/699853</a>
      — the &ldquo;shift &amp; stretch&rdquo; matching intuition (horizontal-well GR snippet → vertical
      reference → &ldquo;you are here&rdquo;).</li>
  <li><a href="https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699326">discussion/699326</a>
      — the offset-typewells insight (three sample wells&apos; typewells visibly the same log with
      depth shifts) and the LWD inversion workflow from the literature.</li>
  <li><a href="https://github.com/hhschumann/LWD_inversion">github.com/hhschumann/LWD_inversion</a>
      — a clean Tikhonov-regularized GR inversion reference. Uses NumPy + Jupyter; ~200 lines.</li>
</ul>

<h2>10. Production Reference: ROGII StarSteer Auto Steer</h2>
<p>ROGII&apos;s commercial geosteering tool, <i>StarSteer</i>, ships an <b>Auto Steer</b> module that
produces exactly the kind of output the competition is asking for: a TVD-vs-MD geological cross-section
with the wellbore overlaid, plus a per-MD-point confidence score. This competition is in effect a public
benchmark for that auto-interpretation task — winning solutions should be benchmarkable against a
real human-quality interpretation produced by the same software.</p>

<h2>11. Practical Pitfalls</h2>
<ul>
  <li><b>Test set is hidden.</b> The 3 visible test wells under <code>test/</code> are only a submission
      template; the real evaluation runs on a hidden set on Kaggle servers. Don&apos;t hard-code well
      IDs; iterate by directory listing.</li>
  <li><b>Validate by leave-one-well-out.</b> Random row splits leak (rows within a well are highly
      auto-correlated). Spatial GroupKFold on XY clusters is the most honest CV.</li>
  <li><b>Handle GR missingness with a mask, not mean-imputation.</b> Median GR missing rate is ~28%
      and the pattern is blocky (long contiguous gaps in some wells).</li>
  <li><b>Don&apos;t leak the formation-top columns at inference.</b> They are training-only; only use
      them as auxiliary supervision.</li>
  <li><b>TVT is not TVD.</b> It is the depth axis of the typewell, not depth-from-surface. The Z column
      is surface-relative elevation and differs from TVT by a per-region datum offset.</li>
</ul>

<h2>12. Suggested Roadmap</h2>
<ol>
  <li>Reproduce <code>baseline_rmse.csv</code> to know your metric floor. Build a working
      <code>submission.csv</code> with the constant baseline first.</li>
  <li>Build a DTW-based predictor with offset-typewell stacking (insight 7.1). Submit; this should
      already be competitive on a sizeable fraction of wells.</li>
  <li>Implement a 1-D Tikhonov inversion (insight in section 5). Initialize it from DTW.</li>
  <li>Wrap that in a sequential filter (section 6) for online stability.</li>
  <li>Train a 1-D CNN / Transformer with the 6 formation-top columns as auxiliary heads (insight 7.2).</li>
  <li>Hybrid: feed the inversion solution + uncertainty as features to the ML model and let it learn
      the residual.</li>
</ol>

<hr/>
<p class="meta">Equations rendered via MathJax. Diagrams generated by <code>report/generate_overview.py</code>.
For the data EDA, see <code>report/eda_report.html</code>.</p>

</body>
</html>
"""

out = REPORT_DIR / "competition_overview.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
