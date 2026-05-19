# `GridMaze/maze/` — maze representations and plotting

Networkx-based graph models of the GridMaze gridworld, plus the plotting + metric helpers that operate on them. Used by both `preprocessing/` (to map pixel coordinates onto maze nodes/edges) and `analysis/` (to compute geodesic distances, plot trajectories, render activity heatmaps).

---

## Modules

| Module | What it provides |
|---|---|
| [`representations.py`](representations.py) | `simple_maze(maze_structure)` and `skeleton_maze(maze_structure)` — networkx graphs built from a session's maze topology. The simple graph keeps the full node/edge layout; the skeleton graph collapses corridors into a connectivity skeleton used for geodesic distance computations. |
| [`metrics.py`](metrics.py) | Graph-derived metrics — geodesic and euclidean distance between maze positions, shortest-path enumeration, allocentric ↔ egocentric direction helpers used throughout the behavioural analyses. |
| [`plotting.py`](plotting.py) | Render maze graphs to matplotlib axes with overlaid heatmaps, trajectory traces, fiber-tip markers, etc. The paper's maze figures and trajectory panels are all built from these helpers. |

---

## Typical use

Via a `MazeSession` (which calls into these modules internally):

```python
from GridMaze.analysis.core import get_sessions as gs

session = gs.get_maze_sessions(
    subject_IDs=["mFC-opto_23"],
    maze_names=["maze_2"],
    days_on_maze=[8],
)[0]

G_simple   = session.simple_maze()      # networkx.Graph
G_skeleton = session.skeleton_maze()    # networkx.Graph
```

Or directly:

```python
from GridMaze.maze import representations as mr

G = mr.simple_maze(session.maze_structure)
```

---

## 🔗 Where to next

- Maze graphs are consumed by behavioural / error analyses → [`analysis/README.md`](../analysis/README.md)
- Maze topology metadata lives in `experiment_info/maze_info.json` — loaded automatically by `get_maze_sessions`
