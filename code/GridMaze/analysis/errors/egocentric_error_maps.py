"""
visualize egocentric error maps,
i.e. where was the goal relative to subject POV when they make navigational errors
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrow

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.errors import errors as be
from GridMaze.analysis.strategies import habits as sh

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %%


def test(
    error_df,
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    emax=3,
    vmin=0,
    vmax=0.05,
    plot_raw=False,
):
    """ """
    # filter data
    df = error_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        # match time in trial data across stim ON/OFF
        df = df[df.time_in_trial <= MAX_STIM_DURATION]
    # get egocentric error heatmap per subject & stim condition
    dfs = []
    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            sub_df = df[(df.subject_ID == subject) & (df.stim_trial == stim_trial)]
            return sub_df
            condition = sub_df.condition.unique()[0]
            heatmap = get_egocentric_error_heatmap(sub_df, emax)
            long_hm = heatmap.stack().reset_index(name="error_rate")
            long_hm["condition"] = condition
            long_hm["stim_trial"] = stim_trial
            dfs.append(long_hm)
    hm_df = pd.concat(dfs, ignore_index=True)

    if plot_raw:
        # plot all group x stim error maps
        f, axes = plt.subplots(2, 2, figsize=(6, 6))
        for i, group in enumerate(["control", "opto"]):
            for j, stim_trial in enumerate([True, False]):
                # cbar = True if i == 1 and j == 1 else False
                plot_df = hm_df[(hm_df.condition == group) & (hm_df.stim_trial == stim_trial)]
                hm = plot_df.groupby(["ego_goal_x", "ego_goal_y"]).error_rate.mean().unstack(1)
                plot_egocentric_error_heatmap(
                    hm,
                    vmin=vmin,
                    vmax=vmax,
                    cbar=True,
                    ax=axes[i, j],
                )
                axes[i, j].set_title(f"{group}: stim={stim_trial}")
        f.tight_layout()
    else:
        # plot Δ error rate (stim - no stim) per group
        f, axes = plt.subplots(1, 2, figsize=(6, 3))
        for group, ax in zip(["control", "opto"], axes):
            plot_df = hm_df[hm_df.condition == group]
            # take difference between stim ON & OFF heatmaps in group
            hm = (
                plot_df.groupby(["ego_goal_x", "ego_goal_y", "stim_trial"])
                .error_rate.mean()
                .unstack(2)
                .diff(axis=1)[True]
                .unstack(0)
            )
            plot_egocentric_error_heatmap(
                hm,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                cbar_label="Δ error rate",
                ax=ax,
            )
            ax.set_title(group)
        f.tight_layout()


def plot_egocentric_error_heatmap(
    hm,
    vmin=0,
    vmax=None,
    cbar=True,
    cbar_label="error rate",
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.heatmap(
        hm.T,
        ax=ax,
        square=True,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        cbar_kws={"label": cbar_label, "shrink": 0.5},
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    # plot white arrow poiting up at 0,0 for extra flavour
    arrow = FancyArrow(
        0.5,
        0.5,  # center of axes
        0.0,
        0.1,  # point north
        head_width=0.05,
        head_length=0.05,
        fc="white",
        ec="white",
        transform=ax.transAxes,  # CRITICAL
        zorder=5,
    )
    ax.add_patch(arrow)


def get_egocentric_error_heatmap(df, emax=3):
    """ """
    # filter for egocentric range
    ego_df = df[df.ego_goal_x.between(-emax, emax) & df.ego_goal_y.between(-emax, emax)]
    ego_grouped = ego_df.groupby(["ego_goal_x", "ego_goal_y"])
    error_rate = ego_grouped.error.sum() / ego_grouped.error.size()
    hm = error_rate.unstack(1)
    return hm
