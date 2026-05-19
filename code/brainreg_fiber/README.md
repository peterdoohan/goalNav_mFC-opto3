# `brainreg_fiber/` — histology registration + fiber-tip localisation

Per-subject pipeline that registers serial-section fluorescence histology to the Allen mouse 10 µm atlas using [`brainreg`](https://brainglobe.info/documentation/brainreg/), then localises each animal's optogenetic fiber tip in atlas coordinates. SLURM job submission for batch processing on an HPC cluster. The brainreg output is consumed by `GridMaze/preprocessing/get_anatomy_data.py`, which produces the per-subject `anatomy/` folder in `processed_data/`.

> ⚠️ **Most users will not run this.** Raw histology stacks are not shipped — the dataset hosted in the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) already contains the downstream per-subject `anatomy_info.json`, `fiber_coordinates.json`, and `registered_signal.tiff` files. This pipeline is documented so the full chain is inspectable.

---

## 🚀 Workflow

The pipeline has three stages — the first and third are automated; the middle one is a manual napari step that cannot be avoided.

### 1. Check brain orientations

Before running brainreg, eyeball one downsampled slice per subject to confirm the orientation string passed to brainreg matches the data:

```python
from brainreg_fiber.run_brainreg import check_brain_orientations
check_brain_orientations()
```

This drops PNGs under `data/preprocessed_data/brainreg/<subject>/orientation_check.png`. Edit `SUBJECT_ID2ORIENTATION` in [`run_brainreg.py`](run_brainreg.py) if any subject needs a non-default orientation (default is `"psl"` for every subject).

### 2. Register all subjects to the Allen 10 µm atlas

```python
from brainreg_fiber import run_brainreg as rb
rb.run_brainreg()
```

This submits one SLURM job per subject (loading the BrainGlobe module on the cluster), each running:

```bash
brainreg <histology_input> <output_dir> \
  --additional <signal_channel> \
  -v <Z> <Y> <X> \
  --orientation <psl> \
  --atlas allen_mouse_10um \
  --debug
```

Channel 3 (green) is used as the registration channel; channel 2 (red) is treated as the signal channel (probe dye). Output lands in `data/preprocessed_data/brainreg/<subject>/allen_mouse_10um/`.

Re-running is safe: subjects whose output directory already exists are skipped (pass `overwrite=True` to force).

### 3. Manually label fiber tips in napari

For each subject:

1. Open the registered `downsampled_standard_2.tiff` in napari.
2. Add a `points` layer.
3. Drop a single point on each fiber tip.
4. Save the points (in atlas voxel space) as a `.csv` file under `<subject>/fiber_coordinates/`.

`GridMaze/preprocessing/get_anatomy_data.py` then reads these CSVs and packages them into the standardised per-subject `fiber_coordinates.json` consumed by the analysis code.

---

## 🧩 Functions

| Function | Purpose |
|---|---|
| [`check_brain_orientations()`](run_brainreg.py) | Saves orientation-check PNGs so you can confirm `SUBJECT_ID2ORIENTATION` matches the raw data before launching brainreg. |
| [`run_brainreg(overwrite=False)`](run_brainreg.py) | Submits one SLURM job per subject to register the histology stack to the Allen 10 µm atlas. |
| [`get_brainreg_paths_df()`](run_brainreg.py) | Returns a dataframe with one row per subject: histology path, brainreg output path, signal/registration channel paths, completion flag, orientation. |
| [`get_brainreg_SLURM_script(br_info)`](run_brainreg.py) | Writes a per-subject SLURM script wrapping the brainreg command. |
| [`get_voxel_sizes(recipe_path)`](run_brainreg.py) | Parses the brainsaw `recipe*.yml` to pull the per-axis voxel sizes brainreg needs. |

---

## 📁 Folders

```
brainreg_fiber/
├── run_brainreg.py          <- main module (functions above)
└── jobs/                    <- created on first run
    ├── slurm/               <- per-subject SLURM scripts
    ├── out/                 <- per-subject stdout
    └── err/                 <- per-subject stderr
```

---

## 🧠 Channel conventions

The brainsaw pipeline saves multiple fluorescence channels:

- **Channel 2** — typically red. The probe / fiber dye channel — used as the **signal** channel passed to brainreg via `--additional`.
- **Channel 3** — typically green. Less autofluorescent than blue, so used as the **registration** channel.
- **Channel 4** — typically blue. Not used.

Edit `SIGNAL_CHANNEL` / `REGISTRATION_CHANNEL` in [`run_brainreg.py`](run_brainreg.py) if your acquisitions use a different mapping.

---

## 🔗 Where to next

- Output is consumed by → [`GridMaze/preprocessing/get_anatomy_data.py`](../GridMaze/preprocessing/get_anatomy_data.py)
- The sibling video-tracking pipeline → [`mazeSLEAP/`](../mazeSLEAP/README.md)
- brainreg documentation → [brainglobe.info/documentation/brainreg](https://brainglobe.info/documentation/brainreg/)
