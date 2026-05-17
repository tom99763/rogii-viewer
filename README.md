# ROGII Viewer

A lightweight desktop geosteering viewer for the
[ROGII — Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
Kaggle competition dataset. Inspired by ROGII's commercial **StarSteer** software, this is a free open-source
tool that lets you browse all 773 training wells (and any number of test wells), inspect the geological
cross-section along each wellbore, correlate gamma-ray signatures against the typewell, and **load any
Kaggle-format predictions CSV to see how your model is doing against the truth, well by well.**

![Main window](viewer/docs/main_window.png)

---

## Features

- **Per-well loader** — scan a dataset folder and browse all train + test wells with search and split filter.
- **Cross-section view** — wellbore Z elevation plotted against MD, with the six formation tops (`ANCC`,
  `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) rendered as color bands so you can immediately see which
  geological layer the lateral is drilling through.
- **TVT prediction view** — overlay truth, `TVT_input`, and your predicted `TVT` on the same MD axis with
  the Prediction Start marker; RMSE is computed instantly when truth is available.
- **GR correlation view** — typewell GR vs TVT alongside the horizontal-well GR; supports overlaying
  predicted-TVT scatter for sanity checking signal alignment.
- **Map view** — XY trajectories of every well, with the active well highlighted and the 8 nearest
  neighbours called out (offset-well exploration).
- **Kaggle-format predictions CSV loader** — drag in any `submission.csv` with columns `id,tvt` (where
  `id = <well_id>_<row_index>`); aligns automatically and shows per-well RMSE.
- **Export to PNG** — Ctrl+E saves the active tab as a PNG.
- **Single-file Windows .exe** — included PyInstaller spec produces a portable `ROGIIViewer.exe`.

---

## Screenshots

### Cross-section with formation-top bands
![Cross-section](viewer/docs/cross_section.png)

### TVT prediction overlay (truth vs `TVT_input` vs prediction)
![TVT prediction](viewer/docs/tvt_prediction.png)

### GR correlation — typewell GR (black) vs horizontal-well GR (green) and prediction (red)
![GR correlation](viewer/docs/gr_correlation.png)

### Map view — every well centroid, with the active well + 8 nearest neighbours
![Map view](viewer/docs/map_view.png)

---

## Quickstart

Pick the path that matches your machine. All three paths share the same prerequisites:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/tom99763/rogii-viewer.git
   cd rogii-viewer
   ```
2. **Download the dataset** (~800 MB, not bundled here):
   ```bash
   kaggle competitions download -c rogii-wellbore-geology-prediction
   unzip rogii-wellbore-geology-prediction.zip -d rogii-wellbore-geology-prediction/
   ```
   The viewer auto-loads `~/ROGII/rogii-wellbore-geology-prediction/` if it exists; otherwise use
   **File → Open dataset folder…** to point at any folder containing `train/` and/or `test/`
   subfolders with `<well_id>__horizontal_well.csv` + `<well_id>__typewell.csv` pairs.

### Path A — Run from source on Linux / WSL / macOS (fastest)

Requires Python 3.10+ and a working display.

```bash
# 1. install dependencies into your active env
pip install -r viewer/requirements.txt

# 2. launch
python -m viewer
```

If the window opens, you're done. **If you hit any Qt error, just use the bundled launcher
script** which sets the right environment variables for you:

```bash
bash viewer/run.sh
```

#### A.1 — If you see a Qt platform plugin error

In **conda environments**, you commonly hit this:

```
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in ""
This application failed to start because no Qt platform plugin could be initialized.
Aborted (core dumped)
```

This means another Qt (typically a conda-installed Qt 6.5/6.6) is being preferred over
PySide6's bundled Qt 6.11, and the version mismatch causes Qt to reject every platform plugin.

**Fix — point Qt at PySide6's bundled plugins explicitly:**

```bash
# one-liner (one-off)
QT_PLUGIN_PATH=$(python -c "import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins'))") \
QT_QPA_PLATFORM=wayland \
python -m viewer
```

**Make it permanent** (so plain `python -m viewer` works):

```bash
cat >> ~/.bashrc <<'EOF'

# ROGII viewer / PySide6 fix: override conda Qt6 with PySide6 bundled Qt
export QT_PLUGIN_PATH="$(python -c 'import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins"))' 2>/dev/null)"
export QT_QPA_PLATFORM=wayland
EOF
source ~/.bashrc
```

> **Notes**
> - `QT_QPA_PLATFORM=wayland` works on **WSLg (Windows 11)** and most modern Linux desktops.
>   If you're on **Windows 10 WSL** (no WSLg), drop the `QT_QPA_PLATFORM` line and install an X
>   server like [VcXsrv](https://sourceforge.net/projects/vcxsrv/), then `export DISPLAY=:0`.
> - If `xcb` is the only option and it complains about missing libs, install them:
>   `sudo apt install -y libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xkb1 libxkbcommon-x11-0`

### Path B — Run from source on Windows (no .exe build needed)

Requires a Windows Python 3.10+ install (download from [python.org](https://www.python.org/downloads/)
and tick *"Add Python to PATH"*).

Open **PowerShell** or **cmd** (not WSL):

```cmd
cd %USERPROFILE%\rogii-viewer
pip install -r viewer\requirements.txt
python -m viewer
```

### Path C — Build a standalone Windows `.exe`

Best for sharing with someone who has no Python install. The build must run on **Windows** — PyInstaller
can only emit a binary for the OS it runs on, so a WSL build will produce a Linux binary, not `.exe`.

In Windows PowerShell or cmd:

```cmd
cd %USERPROFILE%\rogii-viewer
viewer\build.bat
```

`build.bat` installs PyInstaller, calls it with the right flags, and produces
`dist\ROGIIViewer.exe` (~80 MB; bundles PySide6 + pyqtgraph + NumPy + pandas). Then either
double-click it in Explorer or run:

```cmd
dist\ROGIIViewer.exe
```

The `.exe` is fully self-contained — copy `dist\ROGIIViewer.exe` to any other Windows machine and
double-click. The first launch takes a few seconds while PyInstaller unpacks its bundle.

---

## Loading a predictions CSV

`File → Load predictions CSV…` accepts the exact format Kaggle expects for this competition:

```
id,tvt
000d7d20_1442,11236.02
000d7d20_1443,11237.05
...
```

`id` decomposes as `<well_id>_<row_index>`. The viewer:

1. Parses and groups by `well_id`.
2. When you click a well, overlays the prediction as a red dotted line on the **TVT Prediction** tab
   and as red X markers on the **GR Correlation** tab.
3. If the active well is a training well (truth available), shows
   `RMSE vs truth: <value> ft` in the right info panel.

This is the easiest way to spot per-well failure modes that an aggregate Kaggle leaderboard score hides.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `python -m viewer` errors with `ModuleNotFoundError: PySide6` | Run `pip install -r viewer/requirements.txt` from the repo root first. If you use conda, make sure you installed into the active env (`which python` should point inside your env). |
| `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"/"xcb"` | Conda Qt6 version-mismatches PySide6's bundled Qt. See **Path A.1** above for the `QT_PLUGIN_PATH` + `QT_QPA_PLATFORM=wayland` one-liner. |
| Window opens but the well list is empty | No dataset detected. Open **File → Open dataset folder…** and pick a folder with `train/` and/or `test/`. |
| Loading is sluggish on first click of a large well | Normal — pyqtgraph caches after the first render. Subsequent clicks are instant. |
| Predictions CSV loads but the right panel says "RMSE: —" | Either no rows in the CSV matched the current well, or you're viewing a test well (no truth available). RMSE only computes for train wells. |
| WSL: window never appears even after the Qt fix | You're on Windows 10 (no WSLg). Use Path B (run on Windows side) or install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) + `export DISPLAY=:0`. |
| `build.bat` errors with `'python' is not recognized` | Windows-side Python isn't on PATH. Reinstall Python from python.org with *"Add Python to PATH"* ticked. |
| `.exe` won't launch — "VCRUNTIME140.dll missing" | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). |

---

## Repository layout

```
.
├── viewer/                      # Desktop viewer (this app)
│   ├── __main__.py              # python -m viewer entry
│   ├── app.py                   # MainWindow + menus + dock panels
│   ├── plots.py                 # CrossSection / GRCorrelation / Map / TVTPrediction widgets
│   ├── data.py                  # DatasetIndex + WellBundle + Predictions loaders
│   ├── smoke_test.py            # offscreen test (CI-friendly)
│   ├── requirements.txt
│   ├── build.bat                # Windows .exe packaging
│   └── docs/                    # screenshots for this README
├── report/                      # Knowledge-base HTML reports (+ CSVs)
│   ├── index.html               # ← start here: consolidated front page
│   ├── eda_report.html          # deep EDA over all 773 wells
│   ├── competition_overview.html# inverse-problem framing + method portfolio (MathJax)
│   ├── generate_index.py        # rebuilds index.html
│   ├── generate_report.py       # rebuilds eda_report.html
│   ├── generate_overview.py     # rebuilds competition_overview.html
│   ├── well_summary.csv         # one row per horizontal well, ~30 features
│   ├── typewell_summary.csv
│   ├── baseline_rmse.csv        # RMSE floors for 3 trivial baselines
│   ├── geology_counts.csv
│   └── test_nearest_neighbours.csv
├── notebook/
│   └── eda-starter.ipynb        # the Kaggle-provided starter notebook
├── README.md                    # ← you are here
├── LICENSE                      # MIT
└── .gitignore
```

---

## The dataset in one paragraph

Each well comes as two CSV files. `<well_id>__horizontal_well.csv` is the lateral well log sampled every
1 ft of measured depth (`MD`), with columns `MD, X, Y, Z, GR, TVT_input` always present; training files
additionally contain `TVT` (the target) and six formation-top elevations (`ANCC, ASTNU, ASTNL, EGFDU,
EGFDL, BUDA`). `<well_id>__typewell.csv` is a paired vertical reference log sampled at 0.5 ft of TVT
with `TVT, GR, Geology`. The goal is to predict `TVT` for every row after the **Prediction Start (PS)**
point — where `TVT_input` becomes `NaN` — typically the last ~73% of each well. Evaluation is RMSE.

For a fuller picture, open `report/eda_report.html` — it covers PS distribution, GR missingness patterns,
baseline RMSE floors per well, the geology vocabulary, and modeling recommendations.

---

## Tech stack

- **GUI**: [PySide6](https://wiki.qt.io/Qt_for_Python) (Qt 6 bindings)
- **Plots**: [pyqtgraph](https://www.pyqtgraph.org/) for GPU-accelerated 2D rendering
- **Data**: NumPy + pandas
- **Packaging**: PyInstaller (`--onefile --windowed`)

---

## Limitations / roadmap

- No real-time survey ingestion or WITSML support — this is a viewer, not a live geosteering platform.
- No 3D view (yet); the cross-section is 2D Z-vs-MD with formation-top bands.
- No DTW or correlation toolkit built into the GUI — those should live in your modeling pipeline.
- Predictions CSV is read-only; the viewer doesn't author submissions.

PRs welcome.

---

## Acknowledgments

- [ROGII](https://www.rogii.com/) for the dataset and for inspiring the viewer's layout via their
  StarSteer product.
- The [Kaggle competition page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
  for the problem framing and the slide deck that ships with the data.

## License

MIT — see [LICENSE](LICENSE).
