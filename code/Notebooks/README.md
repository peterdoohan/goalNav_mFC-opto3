<div align="center">

# 📓 Notebooks

</div>

The main entry point to the opto analyses in the companion paper. The numbered notebook (`2.opto.ipynb`) reproduces the figures from the opto section (Figure 2) of the paper; the `_extra` notebook holds the supplementary analyses that accompany it.

---

## 🔁 General workflow

Each notebook follows the same recipe:

1. **Set the working directory** to the `code/` folder so the `GridMaze` package is importable:
    ```python
    import os
    os.chdir("/path/to/parent_folder/code")
    ```
2. **Import** the relevant `GridMaze` analysis modules.
3. **Call** functions that load `processed_data/` and `analysis_data/` via [`GridMaze/analysis/core/get_sessions.py`](../GridMaze/analysis/core/get_sessions.py).
4. **Run** the analysis.
5. **Plot** the summary figures.

Figures are saved to `../results/figures/<notebook_name>/`.

> 💡 If you are interested in a specific analysis from the paper, open the notebook and scroll to the relevant cell. The `GridMaze` functions being called are the analysis implementation — **Ctrl/Cmd-click the import in any IDE (or GitHub)** to jump straight to the source.

---

## 📚 Directory

| Notebook | Paper figure | Contents |
|---|---|---|
| [`2.opto.ipynb`](2.opto.ipynb) | Fig. 2 | Learning curves and expert performance (opto vs. control, per maze); within-trial performance under mFC inhibition (excess steps, distance-to-goal on stim vs. non-stim trials); allocentric / egocentric error maps; error-correction dynamics; mixture-of-strategies model fits and opto effects on strategy weights. |
| [`opto_extra.ipynb`](opto_extra.ipynb) | Supplementary | Per-subject viral-expression anatomy (GtACR2 fluorescence inside the fiber-tip illumination cone, broken down by frontal-cortex region); control-group performance sanity checks; expression-vs-behavioural-effect correlations. |

---

## ⚙️ Running notebooks yourself

Make sure you have:

1. Set up this repo next to `data/` and `results/` folders (see [main README](../../README.md)).
2. Downloaded `processed_data/` and `experiment_info/` from the [companion data repo](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA) — either via `bash download_data.sh` from inside `code/`, or by cloning the data repo and following its README.
3. Populated `analysis_data/`:
   ```python
   from GridMaze.analysis.processing import get_movement_threshold as gmt
   from GridMaze.analysis.processing import populate_analysis_data as pad
   gmt.save_movement_threshold()
   pad.populate_analysis_data(parallel_jobs=8)
   ```
   See [`GridMaze/analysis/processing/README.md`](../GridMaze/analysis/processing/README.md) for the full recipe.
4. By default `GridMaze/paths.py` resolves `data/` and `results/` relative to `code/` — no edits needed. Only edit it if you placed data somewhere else.
5. Try running the notebooks — [raise a GitHub issue](https://github.com/peterdoohan/GridMaze-mFC-opto/issues) if things go wrong!

---

## 💻 Related code

- [`GridMaze/`](../GridMaze/README.md) — `get_maze_sessions` API, processed-data format spec
- [`GridMaze/analysis/`](../GridMaze/analysis/README.md) — behaviour / errors / strategies / anatomy subpackage map
- [`GridMaze/analysis/processing/`](../GridMaze/analysis/processing/README.md) — `analysis_data/` generation recipes

---

## 🔗 Related repos

- ⚡ **Ephys experiment code and results** — [github.com/peterdoohan/GridMaze-mFC](https://github.com/peterdoohan/GridMaze-mFC)
- 📊 **Companion data repo** — [github.com/peterdoohan/GridMaze-mFC-opto-DATA](https://github.com/peterdoohan/GridMaze-mFC-opto-DATA)
