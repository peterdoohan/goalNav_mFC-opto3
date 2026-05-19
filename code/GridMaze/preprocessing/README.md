# `GridMaze/preprocessing/` — raw → `processed_data/`

Converts the rig outputs (pyControl behavioural logs, SLEAP-tracked video, brainreg-registered histology) into the standardised, human-readable per-session files described in the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA).

> ⚠️ **Most users will not run this.** Raw and preprocessed tiers (`data/raw_data/`, `data/preprocessed_data/`) are not shipped — they are too large to host publicly. The pipeline is documented here so the full chain is inspectable. To work with the dataset, download the already-built `data/processed_data/` from the data repo.

---

## 🚀 Entry point

```python
from GridMaze.preprocessing import populate_processed_data as ppd

ppd.populate_processed_data()
```

`populate_processed_data` loops over every session listed in the sessions data directory, runs each data-stream processor, and writes one folder per session into `data/processed_data/<subject_ID>/<date>.maze/`. It then runs the per-subject anatomy pipeline.

### Signature

```python
def populate_processed_data(
    session_data_streams=["pycontrol", "session_info", "video"],
    populate_anatomy_data=True,
    anatomy_data_structures=["registered_signal", "fiber_coordinates", "anatomy_info"],
    subject_IDs="all",
    session_dates="all",
    overwrite=False,
)
```

Re-running with `overwrite=False` (the default) skips files that already exist, so you can resume after a crash without redoing finished work.

---

## 📦 What ends up in each session folder

| File | Built by | Contents |
|---|---|---|
| `session_info.json` | [`get_session_info.py`](get_session_info.py) | Subject, date, condition (`opto`/`control`), stim flag, maze name, day-on-maze, experiment phase, rig, tethered flag |
| `events.htsv` | [`get_pycontrol_dfs.py`](get_pycontrol_dfs.py) | All pyControl events (cues, rewards, port pokes, laser on/off) with timestamps |
| `trials.htsv` | [`get_pycontrol_dfs.py`](get_pycontrol_dfs.py) | Per-trial: goal, stim_trial flag, cue/reward times, error pokes |
| `frames.tracking.htsv` | [`get_frames_dfs.py`](get_frames_dfs.py) | Raw SLEAP bodypart positions (one row per video frame) |
| `frames.trajectories.htsv` | [`get_frames_dfs.py`](get_frames_dfs.py) + [`get_maze_trajectories.py`](get_maze_trajectories.py) + [`get_head_direction.py`](get_head_direction.py) + [`sleap_qc.py`](sleap_qc.py) | QC'd centroid trajectory + head direction in maze coordinates |
| `frames.trialInfo.htsv` | [`get_frames_dfs.py`](get_frames_dfs.py) | Per-frame trial / trial-phase / goal / `stim_on` annotations |

And per-subject in `processed_data/<subject_ID>/anatomy/`:

| File | Built by | Contents |
|---|---|---|
| `anatomy_info.json` | [`get_anatomy_data.py`](get_anatomy_data.py) | Viral vector, target region, fiber count, hemisphere |
| `fiber_coordinates.json` | [`get_anatomy_data.py`](get_anatomy_data.py) | Left/right fiber tip coords in Allen-atlas voxel space |
| `registered_signal.tiff` | [`get_anatomy_data.py`](get_anatomy_data.py) | Fluorescence channel registered to the Allen mouse atlas |

---

## 🧩 Modules

The pipeline is split into focused modules. Most are called from `populate_processed_data.py`; you usually don't import them directly.

| Module | Responsibility |
|---|---|
| [`populate_processed_data.py`](populate_processed_data.py) | Top-level driver. Iterates sessions and dispatches to the per-stream populators. |
| [`get_session_info.py`](get_session_info.py) | Builds `session_info.json` for one session — pulls condition, stim, maze, day-on-maze from `experiment_info/` metadata. |
| [`get_pycontrol_dfs.py`](get_pycontrol_dfs.py) | Parses pyControl `.txt` logs → `events.htsv` and `trials.htsv`. |
| [`get_frames_dfs.py`](get_frames_dfs.py) | Reads SLEAP `.h5` predictions, syncs to pyControl time, builds the three `frames.*.htsv` tables. |
| [`get_maze_trajectories.py`](get_maze_trajectories.py) | Maps pixel-space SLEAP points → maze-space trajectories (which node/edge the animal is on). |
| [`get_head_direction.py`](get_head_direction.py) | Estimates head direction from multi-bodypart SLEAP output. |
| [`maze_registration.py`](maze_registration.py) | Per-session pixel ↔ maze-position registration. |
| [`pixels_to_position.py`](pixels_to_position.py) | Geometric helper that applies a registration to convert pixel coords → maze coords. |
| [`sleap_qc.py`](sleap_qc.py) | Quality control on SLEAP outputs (interpolation, outlier rejection). |
| [`get_anatomy_data.py`](get_anatomy_data.py) | Per-subject anatomy: reads manually-labelled fiber-tip CSVs and brainreg outputs, writes `anatomy/` folder. |
| [`get_data_directory.py`](get_data_directory.py) | Walks `raw_data/` + `preprocessed_data/` and returns a dataframe of session → file paths for the driver to iterate. |
| [`rsync.py`](rsync.py) | Helper for syncing files between the rig machine and the analysis machine. |

---

## 🔗 Hand-offs to sidecar pipelines

Two preprocessing steps live outside this directory because they depend on heavy GPU / atlas dependencies that aren't installed in the default environment:

- 🐭 **SLEAP video tracking** — [`mazeSLEAP/`](../../mazeSLEAP/README.md). Takes raw `.mp4` videos in `data/raw_data/video/` → `.h5` predictions in `data/preprocessed_data/SLEAP/`. Consumed by [`get_frames_dfs.py`](get_frames_dfs.py).
- 🧠 **Brainreg histology registration** — [`brainreg_fiber/`](../../brainreg_fiber/README.md). Takes serial-section histology in `data/raw_data/histology/` → atlas-registered stacks in `data/preprocessed_data/brainreg/`. Consumed by [`get_anatomy_data.py`](get_anatomy_data.py).

Run those two first, then `populate_processed_data()`.

---

## 🔗 Where to next

- Full processed-data format spec → [`GridMaze/README.md`](../README.md)
- Generating `analysis_data/` from `processed_data/` → [`analysis/processing/README.md`](../analysis/processing/README.md)
- Sidecar pipelines: [`mazeSLEAP/`](../../mazeSLEAP/README.md) · [`brainreg_fiber/`](../../brainreg_fiber/README.md)
