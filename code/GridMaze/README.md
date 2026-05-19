# `GridMaze/` — core package

The Python package containing preprocessing, analysis, and maze-representation code for the GridMaze opto experiment. This README is the single documentation entry-point for everything inside `GridMaze/`. The one exception is [`analysis/processing/README.md`](analysis/processing/README.md), a deep reference for the `analysis_data/` generation pipeline.

---

## Quick map

```
GridMaze/
├── paths.py              <- DATA_PATH / RESULTS_PATH definitions
├── preprocessing/        <- raw → processed_data ETL (most users don't run this)
├── maze/                 <- maze graph representations + plotting
└── analysis/
    ├── core/             <- session-loading API (get_maze_sessions)
    ├── processing/       <- generates analysis_data/ (deep reference: ./analysis/processing/README.md)
    └── <themed sublibraries>  <- the paper analyses (behaviour, errors, strategies, anatomy)
```

---

## `paths.py`

Defines the data + results paths used by every downstream module. Defaults resolve relative to the `code/` directory, so the `parent_folder/{code, data, results}` layout works out of the box:

```python
PROCESSED_DATA_PATH = Path("../data/processed_data")
ANALYSIS_DATA_PATH  = Path("../data/analysis_data")
RESULTS_PATH        = Path("../results")
```

Change these lines if your data lives elsewhere — see [Configuring paths](../../README.md#configuring-paths-only-if-needed) in the main README.

---

## Loading data: `get_maze_sessions`

The entry point for every analysis. Defined in [`analysis/core/get_sessions.py`](analysis/core/get_sessions.py). Filters sessions by metadata and returns `MazeSession` objects pre-loaded with requested data.

```python
from GridMaze.analysis.core import get_sessions as gs
```

### Signature

```python
def get_maze_sessions(
    subject_IDs="all",
    conditions="all",           # "opto", "control"
    big_maze_rig="all",         # "L", "R"
    sex="all",                  # "male", "female"
    maze_order="all",           # 1, 2
    maze_names="all",           # "maze_1", "maze_2"
    days_on_maze="all",
    experiment_phases="all",    # "learning", "expert"
    total_stim_days="all",
    stim_only=False,
    tethered_only=False,
    experimental_days="all",
    with_data="all",
    must_have_data=True,
    verbose=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `subject_IDs` | `str` or `list[str]` | `"all"` | `"all"` loads every subject in `experiment_info/subject_IDs.json`; or pass a list like `["mFC-opto_23", "mFC-opto_31"]`. |
| `conditions` | `str` or `list[str]` | `"all"` | Filter by viral condition: `"opto"` (stGtACR2) or `"control"` (mCherry). |
| `big_maze_rig` | `str` | `"all"` | Filter by rig: `"L"` or `"R"` (subjects were counterbalanced across two physical maze rigs). |
| `sex` | `str` | `"all"` | `"male"` or `"female"`. |
| `maze_order` | `int` or list | `"all"` | First or second maze the subject experienced (counterbalanced across conditions). |
| `maze_names` | `str` or `list[str]` | `"all"` | `"maze_1"` or `"maze_2"`. |
| `days_on_maze` | `int` or list | `"all"` | 1-indexed day count for this animal on this maze. |
| `experiment_phases` | `str` or list | `"all"` | `"learning"` or `"expert"`. |
| `stim_only` | `bool` | `False` | If `True`, only sessions where the laser was on. |
| `tethered_only` | `bool` | `False` | If `True`, only sessions after fiber tethering. |
| `with_data` | `str` or `list[str]` | `"all"` | `"all"` loads every available `processed_data/` and `analysis_data/` attribute. |
| `must_have_data` | `bool` | `True` | If `True`, sessions missing any requested `with_data` attribute are dropped from the result. |
| `verbose` | `bool` | `True` | If `True`, prints a warning each time a requested file is missing. |

### Return type

Returns `list[MazeSession]`. An empty list (with a warning) if no sessions match.

### `MazeSession` — metadata attributes

Always populated, copied verbatim from `session_info.json`:

| Attribute | Type | Description |
|---|---|---|
| `subject_ID` | `str` | e.g. `"mFC-opto_23"` |
| `condition` | `str` | `"opto"` or `"control"` |
| `date` | `datetime.date` | session date |
| `maze_name` | `str` | `"maze_1"` or `"maze_2"` |
| `maze_order` | `int` | which maze this was for this animal (1 or 2) |
| `day_on_maze` | `int` | 1-indexed day count |
| `stim` | `bool` | `True` if the laser was on this session |
| `tethered` | `bool` | `True` if the fiber was connected this session |
| `experiment_phase` | `str` | `"learning"` or `"expert"` |
| `name` | `str` | `"{subject_ID}.{date}.{session_type}"` |
| `maze_structure` | `dict` | maze topology (nodes, edges, layout) |

### `MazeSession` — data attributes

**Loaded from `processed_data/`** (shipped via the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA)):

| Attribute | Source file |
|---|---|
| `session_info` | `session_info.json` |
| `trials_df` | `trials.htsv` |
| `events_df` | `events.htsv` |
| `tracking_df` | `frames.tracking.htsv` |
| `trajectories_df` | `frames.trajectories.htsv` |
| `trial_info_df` | `frames.trialInfo.htsv` |

**Loaded from `analysis_data/`** (generated locally via `populate_analysis_data`):

| Attribute | Source file |
|---|---|
| `navigation_df` | `frames.navigation.parquet` |
| `trajectory_decisions_df` | `trajectory_decisions_dataframe.parquet` |

> Missing files → the attribute exists but is `None`. Pass `must_have_data=True` to drop sessions that are missing any requested attribute.

### `MazeSession` — methods

- `simple_maze()` — networkx graph of the maze with simplified topology.
- `skeleton_maze()` — networkx graph of just the connectivity skeleton.

---

## `analysis/`

The paper analyses, split into themed subpackages. Each is roughly one analysis topic.

| Subpackage | Computes |
|---|---|
| [`core/`](analysis/core/) | Session loading API (`get_maze_sessions`), `MazeSession`, IO helpers, anatomy loader, shared plotting |
| [`processing/`](analysis/processing/README.md) | Builds `analysis_data/` parquets — see its own README |
| `behaviour/` | Learning curves, expert performance, distance-to-goal, excess steps, movement, trajectory plotting |
| `errors/` | Allocentric / egocentric error maps, error-correction trials |
| `strategies/` | Mixture-of-strategies behavioural model fits, opto effects on strategies, psychometrics |
| `anatomy/` | Per-subject viral-expression blobs, fiber-coordinate localisation, anatomy-vs-behaviour correlations |

For the figure-by-figure walkthrough that calls into these subpackages, see [`Notebooks/README.md`](../Notebooks/README.md).

---

## `maze/`

Maze representations and plotting helpers — networkx-based graph models of the gridworld plus utilities for rendering activity onto them.

- `representations.py` — build `simple_maze` and `skeleton_maze` networkx graphs from session maze structure
- `metrics.py` — graph-derived metrics (shortest paths, geodesic / euclidean distance, allocentric / egocentric helpers)
- `plotting.py` — render maze graphs with overlaid activity heatmaps and trajectory traces

Typical use is via a `MazeSession` (which calls the same functions internally):

```python
session = gs.get_maze_sessions(subject_IDs=["mFC-opto_23"], maze_names=["maze_2"], days_on_maze=[8])[0]
G = session.simple_maze()        # networkx graph
```

---

## `preprocessing/`

The code is here so the full preprocessing chain is inspectable, but the raw data to generate it is not provided.

```python
from GridMaze.preprocessing import populate_processed_data as ppd
ppd.populate_processed_data()
```

Relies on metadata files in `experiment_info/` (`maze_info.json`, `subject_IDs.json`, `fiber_tethered_date.json`, …) and on the sidecar pipelines [`mazeSLEAP/`](../mazeSLEAP/README.md) and [`brainreg_fiber/`](../brainreg_fiber/README.md) to have produced `preprocessed_data/` first.

### `processed_data/` conventions

Per-session folder named `<YYYY-MM-DD>.maze`. Filenames follow the [IBL convention](https://doi.org/10.1101/827873) `object.attribute.filetype`:

- All files for the same `object` share the same first dimension — e.g. every `frames.*` file has one row per video frame.
- `.htsv` for tabular data with tab separation and a header row (load with `pandas.read_csv(..., sep="\t")`).
- `.json` for nested key/value metadata.

**Units:** times are in seconds from pyControl session start; spatial measurements in SI units.

**File list per session:** `session_info.json`, `events.htsv`, `trials.htsv`, `frames.tracking.htsv`, `frames.trajectories.htsv`, `frames.trialInfo.htsv`, plus a per-subject `anatomy/` folder containing `anatomy_info.json`, `fiber_coordinates.json`, and `registered_signal.tiff`.

For the full per-field tour, see the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) README — and the deep preprocessing-pipeline reference at [`preprocessing/README.md`](preprocessing/README.md).

---

## Where to next

- Reproducing a paper figure → [`Notebooks/README.md`](../Notebooks/README.md)
- Generating `analysis_data/` → [`analysis/processing/README.md`](analysis/processing/README.md)
- Raw-data preprocessing details → [`preprocessing/README.md`](preprocessing/README.md)
- Paper context, env setup, repo layout → [main README](../../README.md)
