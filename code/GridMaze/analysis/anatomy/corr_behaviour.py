"""
Correlate features from inhibited anatomy with behaviour deficits.
"""

# %% Imports
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from GridMaze.analysis.anatomy import blobs as ab
from GridMaze.analysis.behaviour import performance_metrics as pm
from GridMaze.analysis.strategies import get_input_data as gid
from GridMaze.analysis.strategies import opto_effects as oe

# %% Global Variables

# %% Functions


def plot_anat_corr_behaviour(
    behavioural_metric="delta_habit",
    regions=["ILA"],
    ratio=["PL"],
    weights_df=None,
    ax=None,
):
    """ """
    # load data from other analyses
    if behavioural_metric == "n_excess_steps":
        # get delta excess steps in opto condition
        performance_metrics = pm.get_performance_df()
        xs_steps_df = pm._get_group_by_stim_df(performance_metrics, y="n_excess_steps")
        _df = xs_steps_df[xs_steps_df.condition == "opto"]
        pivot_df = _df.pivot(index="subject_ID", columns="stim_trial", values="n_excess_steps")
        behav_metric = pivot_df[True] - pivot_df[False]
        metric_label = "Δ Excess steps"

    elif behavioural_metric in ["delta_habit", "delta_structure"]:
        # run standard mix strats analysis to get weights
        if weights_df is None:
            strategies = ["vector", "structure", "habit", "backtracking_penalty"]
            navigation_strategies_df = gid.get_navigation_strategies_df(strategies=strategies, close_far_cutoff=4)
            weights_df = oe.get_group_by_stim_strategy_weights(
                navigation_strategies_df,
                strategies=strategies,
                stim_day_range=(6, np.inf),
                stim_only=False,
            )
        strat = behavioural_metric.split("_")[1]
        _df = weights_df[weights_df.condition == "opto"]
        pivot_df = weights_df.pivot(index="subject_ID", columns="stim_trial", values=[strat])[strat]
        behav_metric = pivot_df[True] - pivot_df[False]
        metric_label = f"Δ {strat}"

    else:
        raise ValueError("behavioural_metric not recognised")

    # derive anat metric
    anatomy_df = ab.get_opto_anatomy_df(condition="opto", verbose=False)
    if regions == "all":
        anat_df = anatomy_df
    else:
        anat_df = anatomy_df[anatomy_df.simple_name.isin(regions)]
    anat_values = anat_df.groupby("subject_ID").voxels.sum()  # convert to um3
    anat_values = anat_values.div(10**3)

    if ratio is None:
        xlabel = "Est. inhibited \n Volume (um$^3$)"
    else:
        # consider ratio between regions
        if ratio == "all":
            anat_2 = anatomy_df
        else:
            anat_2 = anatomy_df[anatomy_df.simple_name.isin(ratio)]
        anat_2_values = anat_2.groupby("subject_ID").voxels.sum()  # convert to um3
        anat_2_values = anat_2_values.div(10**3)
        anat_values = anat_values.div(anat_2_values)

        anat_values.dropna(inplace=True)
        xlabel = "Est. inhibited Volume \n Ratio"
        if len(regions) == 1 and len(ratio) == 1:
            xlabel += f" ({regions[0]} / {ratio[0]})"

    subject_order = anat_values.index

    # set up figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label)

    # plot scatter
    x = anat_values.loc[subject_order]
    y = behav_metric.loc[subject_order]
    ax.scatter(x, y, s=30, color="k", alpha=0.7)

    # plot fit
    coef = np.polyfit(x, y, 1)
    ax.plot(
        x,
        np.polyval(coef, x),
        color="royalblue",
        lw=2,
    )

    # stats
    rho, p = spearmanr(x, y)
    # ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title(f"ρ={rho:.2f}, p={p:.3g}", fontsize=8)
