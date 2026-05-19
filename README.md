<div align="center">

### Code repository for *Doohan et al., 2026* [optogenetics experiment]

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-BSD--style-green)

<pre>
●     ●─────●     ●     ●─────●─────●
│     │     │     │     │     │      
●─────●     ●     ●─────●     ●     ●
│           │     │           │     │
●─────●─────●─────●     ●─────●─────●
│           │     │     │     │     │
●     ●─────●   ⚡🐭⚡    ●     ●     ●
│           │     │           │      
●─────●     ●     ●─────●─────●─────●
│           │           │     │     │
●     ●─────●─────●     ●     ●     ●
│     │                 │     │     │
●─────●─────●─────●─────●     ●     ●
</pre>

# Structured and flexible representations in medial-frontal cortex supports goal-directed navigation

</div>

---

> 🚧 **Work in progress** 🛠️ — code in this repository is actively maintained and subject to change ahead of final publication of the accompanying manuscript. If you encounter any issues, please [open a GitHub issue](https://github.com/peterdoohan/GridMaze-mFC-opto/issues) and I'll get back to you as soon as I can.

> 📊 **Just want the data, not these analyses?** A lightweight companion repo at [github.com/peterdoohan/GridMaze-mFC-opto-DATA](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) hosts the same dataset and a loader package without this analysis codebase — recommended starting point if you're interested in the dataset but not the specific analyses in our paper.

---

**This repo contains code for:**
- **Preprocessing** raw behavioural, video, and histology data into the GridMaze standard format
- **Analysis** codebase implementing all optogenetic-experiment analyses presented in the parent paper
- **Notebook summaries** of those analyses
- **SLEAP video-tracking pipeline** ([SLEAP](https://sleap.ai/)) for top-down pose estimation
- **Fiber-tip localisation pipeline** ([brainreg](https://brainglobe.info/documentation/brainreg/)) for registering histology to the Allen mouse atlas

---

## 📁 Project organisation

This code repo is designed to live inside a parent folder alongside its `data/` and `results/` directories:

```
parent_folder/
├── 💻 code/      <- this repo
├── 📦 data/      <- raw, preprocessed, processed, and analysis data
└── 📈 results/   <- figures and saved analysis outputs
```

The sections below walk through downloading the data, setting up the environment, and running the code in that order.

---

## 📥 Downloading data and results

The processed dataset and per-subject anatomy lives in the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA). Two ways to grab it:

#### Option 1 — `download_data.sh` helper script

From inside `code/`:

```bash
bash download_data.sh
```

This lands `processed_data/` and `experiment_info/` in `../data/`, so the default `paths.py` resolution works without further configuration.

#### Option 2 — Manual

Clone the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) and follow its README. Place the resulting `data/` folder next to this `code/` folder:

```
parent_folder/
├── code/      <- this repo
└── data/      <- from GridMaze-mFC-opto-DATA
```

> 📥 **Raw + preprocessed data are not shipped.** Only `processed_data/` and `experiment_info/` are available for download — the raw video, pyControl logs, and serial-section histology are too large to host publicly. The preprocessing code is still provided so the full chain is inspectable. See [`GridMaze/preprocessing/README.md`](code/GridMaze/preprocessing/README.md).

---

## ⚙️ Environment installation

The Python environment is managed with [miniconda](https://docs.conda.io/projects/miniconda/). Once miniconda is installed:

```bash
git clone https://github.com/peterdoohan/GridMaze-mFC-opto.git code
cd code
conda env create -f environment.yml
conda activate GridMaze_mFC_opto
pip install -e .
```

This installs Python 3.12 and the pinned set of dependencies used across preprocessing, analysis, and notebooks. Tested on Linux.

> 🧠 **Optional:** the SLEAP tracking (`mazeSLEAP/`) and brainreg pipelines (`brainreg_fiber/`) pull in heavy GPU / atlas dependencies. Only install these if you need to re-run preprocessing from raw data — see the individual READMEs in [`mazeSLEAP/`](code/mazeSLEAP/README.md) and [`brainreg_fiber/`](code/brainreg_fiber/README.md).

---

## 📦 Data organisation

```
data/
├── raw_data/                <- data as it comes off the rig (not shipped)
│   ├── pycontrol/           <- behavioural task readout
│   ├── video/               <- top-down video of animals on the maze
│   └── histology/           <- serial-section fluorescence stacks
├── preprocessed_data/       <- outputs from raw-data preprocessing (not shipped)
│   ├── SLEAP/               <- top-down pose tracking
│   └── brainreg/            <- histology registered to the Allen atlas
├── processed_data/          <- standardised, human-readable data (subject_ID/session_ID/)
├── analysis_data/           <- derived analysis tables (generated locally)
└── experiment_info/         <- subject IDs, dates, maze configurations, etc.
```

For the full processed-data format spec — file types, naming conventions, units — see [`code/GridMaze/README.md`](code/GridMaze/README.md). If you only want the dataset without this codebase, the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) is a friendlier starting point.

---

## 💻 Code organisation

```
code/
├── GridMaze/                <- core package: preprocessing + analysis for GridMaze-opto
│   ├── preprocessing/       <- maps raw data → standardised processed_data
│   ├── analysis/            <- generates all analyses presented in the paper
│   ├── maze/                <- networkx-based maze representations + plotting
│   └── paths.py             <- central registry of data + results paths
├── Notebooks/               <- main entry point — analyses by paper section
├── mazeSLEAP/               <- SLEAP top-down video tracking pipeline (raw video → SLEAP/)
└── brainreg_fiber/          <- histology registration + fiber-tip localisation (histology → brainreg/)
```

Per-folder READMEs cover specific subsystems in more detail:
[`Notebooks/`](code/Notebooks/README.md) · [`GridMaze/`](code/GridMaze/README.md) · [`GridMaze/analysis/processing/`](code/GridMaze/analysis/processing/README.md) · [`mazeSLEAP/`](code/mazeSLEAP/README.md) · [`brainreg_fiber/`](code/brainreg_fiber/README.md).

---

## ▶️ Running code locally

After downloading `processed_data/`, you'll need to generate `analysis_data/` from it — handled by [`GridMaze/analysis/processing`](code/GridMaze/analysis/processing/README.md). `analysis_data/` mirrors `processed_data/` in structure but contains derived data tables that are convenient starting points for the analyses in the Notebooks.

**Build `analysis_data/`** from `processed_data/`:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
pad.populate_analysis_data()
```

> ⚠️ Slow to generate without multiprocessing. See [`GridMaze/analysis/processing/README.md`](code/GridMaze/analysis/processing/README.md) for the full recipe.

**If you're interested in how `processed_data/` was generated from raw recordings** (raw + preprocessed data are not shipped, so most users skip this): see [`GridMaze/preprocessing/README.md`](code/GridMaze/preprocessing/README.md).

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

### Configuring paths (only if needed)

By default `GridMaze/paths.py` resolves `data/` and `results/` relative to `code/`, so the layout above just works. If you placed data elsewhere, edit:

```python
# code/GridMaze/paths.py
PROCESSED_DATA_PATH = Path("/absolute/path/to/your/data/processed_data")
ANALYSIS_DATA_PATH  = Path("/absolute/path/to/your/data/analysis_data")
RESULTS_PATH        = Path("/absolute/path/to/your/results")
```

> ℹ️ Scripts and notebooks assume CWD = `code/` (notebooks `os.chdir` to it automatically). If you run a script from a different working directory, prefer absolute paths in `paths.py`.

---

## 📓 Jumping into the analyses

The `Notebooks/` folder is the main entry point for reproducing each opto figure of the paper.

See [`Notebooks/README.md`](code/Notebooks/README.md) for the notebook index.

> 💡 New to the dataset? The [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) hosts simpler walked-through notebooks aimed at general data exploration, not paper-specific analyses.

---

## 🔗 Related repos

- ⚡ **Ephys experiment code and results** — [github.com/peterdoohan/GridMaze-mFC](https://github.com/peterdoohan/GridMaze-mFC)
- 📊 **Companion data repo** — [github.com/peterdoohan/GridMaze-mFC-opto-DATA](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA)

---

## 📜 Citation

```bibtex
@article{placeholder,
  title  = {Structured and flexible representations in medial-frontal cortex
            support goal-directed navigation},
  author = {Doohan, Peter T. Jensen, Jensen, Kristopher, T. Chen, Yaqing. Godinho, Beatriz. Burns, Charles D.G. Qin, Chongyu (Xiao). Emery, Josie. Cini, Ryan. Walton, Mark E. T. Behrens, Timothy E.J. Akam, Thomas E.},
  year   = {2026}
}
```

---
