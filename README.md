<div align="center">

### Code repository for *Doohan et al., 2026* [optogenetics experiment]

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-BSD--style-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20268950.svg)](https://doi.org/10.5281/zenodo.20268950)

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
- **Viral expression localaisation pipeline** ([brainreg](https://brainglobe.info/documentation/brainreg/)) for registering histology to the Allen mouse atlas

---

## 📁 Project organisation

This code repo is designed to live inside a parent folder alongside its `data/` and `results/` directories:

```
parent_folder/
├── 💻 code/      <- this repo
├── 📦 data/      <- processed data, exepriment info and analysis data
└── 📈 results/   <- figures and saved analysis outputs
```

The sections below walk through downloading the data, setting up the environment, and running the code in that order.

---

## 📥 Downloading data and results

Both `data/` and `results/` are archived on Zenodo:

- 🆔 [10.5281/zenodo.20268950](https://doi.org/10.5281/zenodo.20268950)

The record contains two zips:

| File          | Contents                                              | Size    |
|---------------|-------------------------------------------------------|---------|
| `data.zip`    | `processed_data/` + `experiment_info/`                | 66.4 GB |
| `results.zip` | saved analysis outputs (figures, behaviour, errors, strategies, anatomy) | 306 MB  |

> 📈 **About `results/`:** all contents of `results/` can be regenerated locally from the code in this repo and `processed_data/` from `data.zip`. `results.zip` is provided for convenience — some saved analyses (permutation tests, strategy-model fits) take a while to recompute, and downloading is faster.

Pick whichever of the three options below suits you. The end goal in every case is the layout:

```
parent_folder/
├── code/      <- this repo
├── data/      <- from data.zip
└── results/   <- from results.zip
```

#### Option 1 — Manual download

1. Open [zenodo.org/records/20268950](https://zenodo.org/records/20268950) in a browser.
2. Download `data.zip` (required) and `results.zip` (optional).
3. Unzip them so the layout above sits next to the cloned `code/`.
4. If you placed `data/` and `results/` somewhere other than next to `code/`, edit `code/GridMaze/paths.py` to point at your local copies (see [Configuring paths](#configuring-paths-only-if-needed) below).

#### Option 2 — curl

From inside `parent_folder/`:

```bash
# required
curl -L -o data.zip    https://zenodo.org/records/20268950/files/data.zip
unzip data.zip && rm data.zip

# optional
curl -L -o results.zip https://zenodo.org/records/20268950/files/results.zip
unzip results.zip && rm results.zip
```

By default `code/GridMaze/paths.py` resolves `data/` and `results/` relative to the `code/` directory, so this layout works out of the box. If you placed the data elsewhere, see [Configuring paths](#configuring-paths-only-if-needed).

#### Option 3 — `download_data.sh` helper script

A helper script in `code/` handles the curl + MD5-verify + unzip dance and lands the data and results in the correct sibling folders, so the default `paths.py` resolution works without further configuration. Run from inside `code/`:

```bash
# defaults: download both data.zip + results.zip, verify MD5, unzip into ../data and ../results
bash download_data.sh

# data only, skip the (small) saved results
bash download_data.sh --no-results

# custom destinations
bash download_data.sh --data-dir /scratch/gridmaze/data --results-dir /scratch/gridmaze/results
```

> 📥 **Raw + preprocessed data are not shared.** Only `processed_data/` and `experiment_info/` are bundled in `data.zip` — the raw video, pyControl logs, and serial-section histology are too large to host publicly. The preprocessing code is still provided so the full chain is inspectable. See [`GridMaze/preprocessing/README.md`](code/GridMaze/preprocessing/README.md).

> 💡 **Lightweight loader only?** If you don't need this analysis codebase, the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) ships a minimal loader package + tutorials over the same Zenodo data.

---

## ⚙️ Environment installation

The Python environment is managed with [miniconda](https://docs.conda.io/projects/miniconda/). Once miniconda is installed:

```bash
git clone https://github.com/peterdoohan/GridMaze-mFC-opto.git code
cd code
conda env create -f environment.yml
conda activate GridMaze_mFC_opto
```

This installs Python 3.12 and the pinned set of dependencies used across preprocessing, analysis, and notebooks. Tested on Linux.


---

## 📦 Data organisation

```
data/
├── raw_data/                <- data as it comes off the rig (not shared)
│   ├── pycontrol/           <- behavioural task readout
│   ├── video/               <- top-down video of animals on the maze
│   └── histology/           <- serial-section fluorescence stacks
├── preprocessed_data/       <- outputs from raw-data preprocessing (not shared)
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

After downloading `processed_data/`, you'll need to generate `analysis_data/` from it — handled by [`GridMaze/analysis/processing`](code/GridMaze/analysis/processing/README.md). `analysis_data/` mirrors `processed_data/` in structure but contains derived data tables that are convenient starting points for the analyses.

**Build `analysis_data/`** from `processed_data/`:

```python
from GridMaze.analysis.processing import populate_analysis_data as pad
pad.populate_analysis_data()
```

> ⚠️ Slow to generate without multiprocessing. See [`GridMaze/analysis/processing/README.md`](code/GridMaze/analysis/processing/README.md) for the full recipe.

**If you're interested in how `processed_data/` was generated from raw recordings** (raw + preprocessed data are not shared directly): see [`GridMaze/preprocessing/README.md`](code/GridMaze/preprocessing/README.md).

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

Please cite both the paper and the dataset:

```bibtex
@article{placeholder,
  title  = {Structured and flexible representations in medial-frontal cortex
            support goal-directed navigation},
  author = {Doohan, Peter T. Jensen, Jensen, Kristopher, T. Chen, Yaqing. Godinho, Beatriz. Burns, Charles D.G. Qin, Chongyu (Xiao). Emery, Josie. Cini, Ryan. Walton, Mark E. T. Behrens, Timothy E.J. Akam, Thomas E.},
  year   = {2026}
}

@dataset{doohan_2026_opto_dataset,
  title     = {Data and results for: Structured and flexible representations in
               medial-frontal cortex support goal-directed navigation
               [optogenetics experiment]},
  author    = {Doohan, Peter T. and Behrens, Timothy E.J. and Akam, Thomas E.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20268950},
  url       = {https://doi.org/10.5281/zenodo.20268950}
}
```

---
