# Analysis data processing

This package turns `processed_data/` (one folder per session) into the compressed, project-specific tables that all downstream analyses (notebooks, paper figures) load from. Everything written here lives in `data/analysis_data/`:

```
analysis_data/
├── analysis_info/                          <- experiment-wide constants (json)
└── <subject_ID>/<session_ID>/              <- per-session analysis tables (parquet)
```

> ❓ **What is analysis data for** ❓ Across analyses we end up doing the same computations many times (eg, deriving navigation features at each video frame, or classifying trajectory decisions). Analysis data is designed to be a convenient starting point for downstream analyses and can be loaded similar to processed data using `MazeSession` objects.

There are two streams:

| Entry point | What it builds | Where it writes |
|---|---|---|
| [`get_movement_threshold.save_movement_threshold()`](get_movement_threshold.py) | Experiment-wide + per-subject speed thresholds for the stationary-vs-moving classification | `analysis_data/analysis_info/movement_threshold.json`, `subject_movement_thresholds.json` |
| [`populate_analysis_data.populate_analysis_data()`](populate_analysis_data.py) | Per-session tables (navigation frames, trajectory decisions) | `analysis_data/<subject>/<session>/*.parquet` |

> ⚠️ **Ordering matters.** `populate_analysis_data` uses the movement threshold to stamp the `moving` column when building `frames.navigation.parquet`. Run `save_movement_threshold()` first.

---

## 🚀 Quick start — generate everything from scratch

After downloading `processed_data/` (see [parent README](../../../../README.md) and the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA)), run this once with CWD set to your `code/` folder:

```python
from GridMaze.analysis.processing import get_movement_threshold as gmt
from GridMaze.analysis.processing import populate_analysis_data as pad

# 1. Fit + save the speed thresholds used by the navigation_df builder.
gmt.save_movement_threshold()

# 2. Build all per-session analysis tables.
#    Slow without multiprocessing — pass parallel_jobs to speed up.
pad.populate_analysis_data(parallel_jobs=8)
```

Each session that finishes gets its parquet files in `analysis_data/<subject>/<session>/`. Re-running with `overwrite=False` (the default) skips files that already exist, so you can resume after a crash without redoing finished work.

### Building only a subset

```python
# one data structure across all sessions
pad.populate_analysis_data(
    data_structures=["navigation_df"],
    parallel_jobs=8,
)

# one subject only
pad.populate_analysis_data(
    subject_IDs=["mFC-opto_23"],
    parallel_jobs=4,
)
```

Valid `data_structures` names: `"navigation_df"`, `"trajectory_decisions_df"`.

---

## 📦 What ends up in `analysis_info/`

| File | Built from | Used by |
|---|---|---|
| `movement_threshold.json` | Per-frame speeds across all sessions, fit with a 2-component GMM in log-space → threshold separating "stationary" from "moving" | Stamps the `moving` column when building `frames.navigation.parquet`; downstream analyses that filter for `moving == True` |
| `subject_movement_thresholds.json` | Same GMM fit, applied per subject | Per-subject behavioural normalisations (movement-tuning controls, opto contrasts) |

Both are written by a single call to `gmt.save_movement_threshold()`.

---

## 📦 What ends up in each `<subject>/<session>/` folder

`populate_analysis_data()` saves one parquet per data structure per session. All are gzipped pandas DataFrames; load them with `pandas.read_parquet`.

### Frame-level table (one row per video frame, 60 Hz)

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `frames.navigation.parquet` | [`get_navigation_dfs.py`](get_navigation_dfs.py) | Trial info, maze position, head direction, velocity/speed, `moving` flag, cardinal movement direction, distance / steps / progress / angle to goal (geodesic, euclidean, future, manhattan, allocentric, egocentric) | Behaviour, distance-to-goal, error-map analyses (notebook 2 + extras) |

### Trajectory table (one row per maze-position visit)

| File | Built by | Contents | Main downstream consumers |
|---|---|---|---|
| `trajectory_decisions_dataframe.parquet` | [`get_trajectory_decisions_dfs.py`](get_trajectory_decisions_dfs.py) | One row per node (and optionally edge) visit, with backtracking corrected, plus action, egocentric action, choice degree, distance and steps to goal | Behavioural strategy fits (`analysis/strategies/`), error analyses (`analysis/errors/`) |

---

## 🧠 Tips, gotchas and likely problems

- **Disk and time.** A full run produces a few GB across all sessions. With `parallel_jobs=8`, expect tens of minutes on a reasonable machine; without it, expect hours.
- **Memory per worker.** Each parallel worker loads one full session's processed-data tables into memory. If you OOM, drop `parallel_jobs`.
- **Skipped sessions are normal.** Sessions in `processed_data/` whose folder name does not end in `.maze` (e.g. open-field sessions) are silently skipped — the prerequisite-check in `_save_navigation_df` / `_save_trajectory_decisions_df` looks for `frames.trajectories.htsv` and `frames.trialInfo.htsv`, which only exist for maze sessions.
- **Missing prerequisite files.** If a session is missing a required processed file, the relevant function returns early with a `Missing pre-requisite processed data structures` log line — the rest of the session continues. Look through the print output to spot sessions that are partially built.
- **Re-running is idempotent.** `populate_analysis_data` defaults to `overwrite=False` and skips any file that already exists. To force a rebuild of a single table, either delete the parquet file or pass `overwrite=True`.

---

## 🔗 Where to next

- Loading sessions with the new analysis tables → [`GridMaze/README.md`](../../README.md#mazesession--data-attributes)
- Reproducing a paper figure that consumes these tables → [`Notebooks/README.md`](../../../Notebooks/README.md)
- Raw → `processed_data/` pipeline → [`preprocessing/README.md`](../../preprocessing/README.md)
