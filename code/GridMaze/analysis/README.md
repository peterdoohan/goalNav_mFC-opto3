# `GridMaze/analysis/` — paper analyses

The analysis codebase that turns `processed_data/` and `analysis_data/` into the figures and statistics for the opto experiment in *Doohan et al., 2026*. For the figure-by-figure walkthrough that calls into these subpackages, see [`Notebooks/README.md`](../../Notebooks/README.md).

---

## 🚀 Entry point

Every analysis begins by loading sessions via `get_maze_sessions`:

```python
from GridMaze.analysis.core import get_sessions as gs

opto_sessions = gs.get_maze_sessions(conditions=["opto"], experiment_phases=["expert"])
```

Returns a list of `MazeSession` objects, each pre-loaded with the requested processed + analysis data. Full filter signature and attribute tables: [`GridMaze/README.md`](../README.md#loading-data-get_maze_sessions).

---

## 🧩 Subpackages

| Subpackage | What it computes |
|---|---|
| [`core/`](core/) | Session-loading API (`get_maze_sessions`, `MazeSession`), data IO (`load_data`), per-subject anatomy loader (`get_anatomy`), and shared plotting helpers |
| [`processing/`](processing/README.md) | Builds the derived `analysis_data/` parquet tables — see its own deep-reference README |
| [`behaviour/`](behaviour/) | Learning curves, expert performance, distance-to-goal, excess-steps, movement, trajectory plotting |
| [`errors/`](errors/) | Allocentric / egocentric error maps, error-correction trial analyses |
| [`strategies/`](strategies/) | Mixture-of-strategies behavioural model fits, habits, psychometrics, opto effects on strategy use |
| [`anatomy/`](anatomy/) | Per-subject viral-expression blobs (fluorescence inside the fiber-tip illumination cone), anatomy-vs-behaviour correlations |

---

## 📂 `core/`

Shared infrastructure that every other subpackage builds on.

| Module | Purpose |
|---|---|
| [`get_sessions.py`](core/get_sessions.py) | `get_maze_sessions(...)` filter + the `MazeSession` class |
| [`load_data.py`](core/load_data.py) | Dispatches to the right loader (`pd.read_csv`, `pd.read_parquet`, `json.load`, …) based on file suffix |
| [`get_anatomy.py`](core/get_anatomy.py) | Per-subject anatomy loader: fiber coords, registered fluorescence, expression breakdown by region |
| [`plotting.py`](core/plotting.py) | Shared matplotlib helpers — paper-wide colour palette, opto-vs-control styling, axis decoration |

---

## 📂 `behaviour/`

Most of the main opto figure lives here.

| Module | What it produces |
|---|---|
| [`learning.py`](behaviour/learning.py) | Per-session trial counts across learning + expert phases. `get_learning_curve_df()`, `get_expert_performance_df()`, `plot_performance_summary(...)`. |
| [`performance_metrics.py`](behaviour/performance_metrics.py) | Per-trial performance metrics for the opto-vs-control comparison on stim vs. non-stim trials. `get_performance_df()`. |
| [`distance_to_goal.py`](behaviour/distance_to_goal.py) | Distance-to-goal traces aligned to trial events; opto vs. control comparisons. |
| [`excess_steps.py`](behaviour/excess_steps.py) | Number of steps taken above the geodesic minimum; the primary behavioural readout in the paper. |
| [`movement.py`](behaviour/movement.py) | Speed, acceleration, stationary-bout statistics; controls for motor confounds. |
| [`trajectory_plotting.py`](behaviour/trajectory_plotting.py) | Plot raw mouse paths overlaid on the maze graph; example trajectories for the figure. |

---

## 📂 `errors/`

Decomposes the steps-above-minimum signal into spatial error patterns.

| Module | What it produces |
|---|---|
| [`errors.py`](errors/errors.py) | Top-level error-trial detection + per-trial error tagging. |
| [`allocentric_error_maps.py`](errors/allocentric_error_maps.py) | Per-node error rate maps in allocentric (maze) coordinates. |
| [`egocentric_error_maps.py`](errors/egocentric_error_maps.py) | Per-decision-point error rate maps in egocentric (relative to heading) coordinates. |
| [`correction.py`](errors/correction.py) | Error-correction dynamics: how quickly the animal recovers an efficient route after a wrong choice. |

---

## 📂 `strategies/`

Mixture-of-strategies model that classifies each decision as random / habitual / goal-directed.

| Module | What it produces |
|---|---|
| [`get_input_data.py`](strategies/get_input_data.py) | Builds the per-decision feature matrix from `trajectory_decisions_dataframe.parquet`. |
| [`models.py`](strategies/models.py) | Strategy-model definitions + fitting code. |
| [`comparisons.py`](strategies/comparisons.py) | Model comparisons (AIC/BIC, cross-validated likelihood). |
| [`habits.py`](strategies/habits.py) | Habit-strategy specific helpers. |
| [`psychometrics.py`](strategies/psychometrics.py) | Strategy-probability psychometrics vs. trial / session covariates. |
| [`opto_effects.py`](strategies/opto_effects.py) | Per-subject opto-vs-control contrasts on fitted strategy weights. |

---

## 📂 `anatomy/`

Localises viral expression in atlas space and ties it to behavioural readouts.

| Module | What it produces |
|---|---|
| [`blobs.py`](anatomy/blobs.py) | `get_opto_anatomy_df(...)` — per-subject fluorescence inside the fiber-tip illumination cone, broken down by frontal-cortex region. `plot_anatomy_summary(...)` for the figure. |
| [`corr_behaviour.py`](anatomy/corr_behaviour.py) | Per-subject correlations between expression amount/location and behavioural opto effect size. |
| [`plotting.py`](anatomy/plotting.py) | Anatomy-specific plotting helpers (atlas overlays, fiber-tip markers). |

---

## 🔗 Where to next

- Reproducing a paper figure → [`Notebooks/README.md`](../../Notebooks/README.md)
- Generating `analysis_data/` → [`processing/README.md`](processing/README.md)
- `MazeSession` API, attribute tables, paths config → [`GridMaze/README.md`](../README.md)
