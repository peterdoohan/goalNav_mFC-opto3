"""
reintroducted excess steps library for follow up analysis on group x stim effects
of increased excess steps during navigation with mFC inhibition
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import zscore, chi2

from GridMaze.maze import representations as mr
from GridMaze.maze import metrics as mm
from GridMaze.maze import plotting as mp

from GridMaze.analysis.behaviour import performance_metrics as pm


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


# %% Functions


# %% old


def test(
    excess_steps_df,
    stim_day_range=(8, np.inf),
    outlier_thres=500,
    var="mean_distance_decorrelation",  # goal fitness
    zscore_var=True,
):
    """ """
    # filter data
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]

    if var in ["start_euclidean_dist", "start_geodesic_dist", "start_dist_ratio"]:
        if var == "start_dist_ratio":
            v = df["start_geodesic_dist"] - df["start_euclidean_dist"]
        else:
            v = df[var]  # already computed

    else:
        # calculate var per trial
        maze_1 = mr.get_simple_maze("maze_1")
        maze_2 = mr.get_simple_maze("maze_2")
        if var == "betweenness_centrality":
            maze_1_dict = mm.get_betweeness_centrality(maze_1)
            maze_2_dict = mm.get_betweeness_centrality(maze_2)
        elif var == "mean_geodesic_distance":
            maze_1_dict = mm.get_mean_shortest_path_distance(maze_1)
            maze_2_dict = mm.get_mean_shortest_path_distance(maze_2)
        elif var == "node_degree":
            maze_1_dict = mm.get_node_degree(maze_1)
            maze_2_dict = mm.get_node_degree(maze_2)
        elif var == "mean_distance_decorrelation":
            maze_1_dict = mm.get_mean_distance_decorrelation(maze_1)
            maze_2_dict = mm.get_mean_distance_decorrelation(maze_2)
        else:
            raise NotImplementedError

        def _map_var(row):
            if row.maze_name == "maze_1":
                return maze_1_dict[row.goal]
            elif row.maze_name == "maze_2":
                return maze_2_dict[row.goal]

        v = df.apply(_map_var, axis=1)

    if zscore_var:
        v = zscore(v)
    df[var] = v

    # fit linear mixed effects model
    md = smf.mixedlm(
        f"n_excess_steps ~ condition * stim_trial * {var}",
        df,
        groups=df["subject_ID"],
        re_formula=f"~stim_trial + {var} + stim_trial:{var}",  # maximum random effects structure
    )
    res = md.fit(reml=False, method="lbfgs", maxiter=10_000)

    # chat gpt suggested model comparison?
    # 2) maximal mixed model (ML for model comparison)
    formula_full = f"n_excess_steps ~ condition * stim_trial * {var}"
    md_full = smf.mixedlm(
        formula_full, df, groups=df["subject_ID"], re_formula=f"~stim_trial + {var} + stim_trial:{var}"
    )
    res_full = md_full.fit(reml=False, method="lbfgs", maxiter=10000)
    print(res_full.summary())

    # 3) reduced model without the 3-way (keep same random structure)
    # remove the 3-way term but keep all lower-order terms
    formula_reduced = f"n_excess_steps ~ (condition * stim_trial) + (condition * {var}) + (stim_trial * {var})"
    md_reduced = smf.mixedlm(
        formula_reduced, df, groups=df["subject_ID"], re_formula=f"~stim_trial + {var} + stim_trial:{var}"
    )
    res_reduced = md_reduced.fit(reml=False, method="lbfgs", maxiter=10000)
    print(res_reduced.summary())

    # 4) LRT (likelihood ratio test) for the 3-way interaction
    llf_full = res_full.llf
    llf_red = res_reduced.llf

    # use number of fixed-effect parameters difference:
    nfe_full = len(res_full.fe_params)  # fixed-effect params in full
    nfe_red = len(res_reduced.fe_params)  # fixed-effect params in reduced
    df_diff = nfe_full - nfe_red

    LR = 2 * (llf_full - llf_red)
    p_val = 1 - chi2.cdf(LR, df_diff)

    print(f"LR = {LR:.3f}, df = {df_diff}, p = {p_val:.4g}")


# %%


def plot_delta_delta_excess_steps_across_goals(
    excess_steps_df,
    maze_name="maze_2",
    stim_day_range=(4, np.inf),
    outlier_thres=500,
    highlight_significant=True,
    vmax=5,
    ax=None,
):
    """ """
    # filter data
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]
    df = df[df.maze_name == maze_name]
    # get delta (light on - light off) per goal per subject
    # then average this delta across subjects in each condition (opto/control)
    delta_df = (
        df.groupby(["subject_ID", "condition", "goal", "stim_trial"])
        .n_excess_steps.mean()
        .unstack(level=-1)
        .diff(axis=1)[True]
        .groupby(level=[1, 2])
        .mean()
        .unstack(level=0)
    )
    delta_delta_df = delta_df[("opto")] - delta_df[("control")]
    # stats (no exaclty correct, just placeholder approx.)
    if highlight_significant:
        # Fit mixed model (with subject random effects)
        md = smf.mixedlm(
            "n_excess_steps ~ condition * stim_trial * goal", df, groups=df["subject_ID"], re_formula="~stim_trial"
        )
        res = md.fit(reml=False, method="lbfgs", maxiter=10_000)
        goal2p_val = {}
        pvals = res.pvalues
        for goal in df.goal.unique():
            try:
                goal2p_val[goal] = pvals.loc[f"condition[T.opto]:stim_trial[T.True]:goal[T.{goal}]"]
            except KeyError:
                pass
        highlight_nodes = [goal for goal, p in goal2p_val.items() if p < 0.05]
    else:
        highlight_nodes = False

    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    simple_maze = mr.get_simple_maze(maze_name)
    edges = [l for l in mr.get_maze_locations(simple_maze) if "-" in l]  # add edge values as 0 for plotting
    mp.plot_simple_heatmap(
        simple_maze,
        pd.concat([delta_delta_df, pd.Series(0, index=edges)]),
        colormap="mako",
        highlight_nodes=highlight_nodes,
        value_label="ΔΔ excess steps",
        highlight_color="magenta",
        node_size=175,
        edge_size=6.5,
        vmin=0,
        vmax=vmax,
        ax=ax,
    )
