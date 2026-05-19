# `mazeSLEAP/` — top-down video tracking

Wrapper around [SLEAP](https://sleap.ai/) for top-down pose tracking of mice on the GridMaze, with SLURM submission scripts for batch processing on an HPC cluster. Reads raw videos from `data/raw_data/video/` and writes per-session `.h5` prediction files to `data/preprocessed_data/SLEAP/`, which `GridMaze/preprocessing/get_frames_dfs.py` then consumes.

> ⚠️ **This code is just for inspection.** Raw video and the resulting SLEAP predictions are not shared. This pipeline is documented so the full chain is inspectable.

---

## 🚀 Entry point

```python
from mazeSLEAP import track_video as tv

# inspect which sessions still need tracking
df = tv.get_video_paths_df()
print(df[~df.tracking_completed])

# submit one SLURM job per untracked session
tv.run_sleap_preprocessing()
```

`run_sleap_preprocessing()` walks every `.mp4` in `data/raw_data/video/`, infers each session's metadata from the filename, picks the right SLEAP model (tethered vs. untethered, plus a retrain for a specific window of R-rig dates), writes a per-session SLURM script under `mazeSLEAP/jobs/slurm/`, and submits it. Outputs land in `data/preprocessed_data/SLEAP/`.

Re-running is safe: sessions already represented in the SLEAP output folder are skipped.

---

## 🧩 Functions

| Function | Purpose |
|---|---|
| [`get_video_paths_df()`](track_video.py) | Returns a dataframe with one row per raw video: `subject_ID`, `datetime`, `tethered`, `video_path`, `tracking_completed`. Used to discover what still needs to be tracked. |
| [`run_sleap_preprocessing()`](track_video.py) | Submits one SLURM job per untracked session. |
| [`get_sleap_SLURM_script(video_info)`](track_video.py) | Writes a per-session SLURM script that calls `track_video(video_path, model_type)` inside the SLEAP conda env. |
| [`track_video(video_path, sleap_model)`](track_video.py) | Loads the video + the matching SLEAP centroid/centered-instance models, runs inference, and saves predictions to `data/preprocessed_data/SLEAP/`. This is what each SLURM job runs. |
| [`load_sleap_predictor(sleap_model)`](track_video.py) | Picks the right pair of model checkpoints out of `mazeSLEAP/models/` and returns a `sleap.Predictor`. |

---

## 📁 Folders

```
mazeSLEAP/
├── track_video.py           <- main module (functions above)
├── models/                  <- SLEAP centroid + centered-instance checkpoints, one set per session type
│   ├── big_maze_opto_untethered.centroid.*           
│   ├── big_maze_opto_untethered.centered_instance.*
│   ├── big_maze_opto_tethered.centroid.*
│   ├── big_maze_opto_tethered.centered_instance.*
│   └── 20-22_fix.{centroid,centered_instance}.*       <- retrain for a specific date window on the R rig
└── jobs/                    <- created on first run
    ├── slurm/               <- per-session SLURM scripts
    ├── out/                 <- per-session stdout
    └── err/                 <- per-session stderr
```

---

## 🧠 Tethered vs. untethered models

Subjects are fiber-tethered from a known date (stored in `data/experiment_info/fiber_tethered_date.json`). The tethered animals look visually different on camera (fiber + patch cable in frame), so we use a separate SLEAP model for tethered sessions. `track_video.run_sleap_preprocessing()` picks the model based on whether the video's date is on or after the tether date.

A small subset of R-rig dates (`2025-11-20` … `2025-11-22`) is re-tracked with a third, retrained model — see `RERUN_DATES` and `R_MAZE_SUBJECTS` in [`track_video.py`](track_video.py).
