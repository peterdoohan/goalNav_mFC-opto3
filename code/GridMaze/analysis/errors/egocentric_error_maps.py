"""
visualize egocentric error maps,
i.e. where was the goal relative to subject POV when they make navigational errors
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from pingouin import mixed_anova
from statsmodels.stats.multitest import multipletests


import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrow
from scipy.ndimage import gaussian_filter

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %%


def plot_egocentric_error_map_summary(
    ego_df,
    wmax=5,
    plot_as="delta_delta",
    colormap="coolwarm",
    vmin=None,
    vmax=None,
    axes=None,
):
    """ """
    df = ego_df.copy()
    if wmax is not None:
        # further restrict window for plotting after smoothing
        df = df[df.ego_goal_x.between(-wmax, wmax) & df.ego_goal_y.between(-wmax, wmax)]
    if plot_as == "raw":
        # plot all group x stim average error maps
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "ego_goal_x", "ego_goal_y"]).error_rate.mean()
        vmax = plot_df.max() if vmax is None else vmax
        vmin = 0 if vmin is None else vmin
        for i, group in enumerate(["control", "opto"]):
            for j, stim_trial in enumerate([False, True]):
                ax = axes[i, j]
                ax.set_title(f"{group}: stim={stim_trial}")
                ego_hm = plot_df.loc[group, stim_trial].unstack(0)
                plot_egocentric_error_heatmap(
                    ego_hm,
                    vmin=vmin,
                    vmax=vmax,
                    colormap=colormap,
                    cbar=True,
                    ax=ax,
                )
    elif plot_as == "delta":
        # get delta stim within subject then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))
        pivot_df = df.pivot_table(
            index=["subject_ID", "condition", "ego_goal_x", "ego_goal_y"],
            columns="stim_trial",
            values="error_rate",
        )
        plot_df = pivot_df.diff(axis=1)[True].groupby(level=[1, 2, 3]).mean()
        vmax = plot_df.max() if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
        for group, ax in zip(["control", "opto"], axes):
            ax.set_title(group)
            plot_ego_hm = plot_df.loc[group].unstack(0)
            plot_egocentric_error_heatmap(
                plot_ego_hm,
                vmin=vmin,
                vmax=vmax,
                colormap=colormap,
                cbar=True,
                cbar_label="Δ error rate",
                ax=ax,
            )
    elif plot_as == "delta_delta":
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            index=["subject_ID", "condition", "ego_goal_x", "ego_goal_y"],
            columns="stim_trial",
            values="error_rate",
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[1, 2, 3]).mean()
        plot_df = delta_df.loc["opto"] - delta_df.loc["control"]
        vmax = plot_df.max() if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
        plot_egocentric_error_heatmap(
            plot_df.unstack(0),
            vmin=vmin,
            vmax=vmax,
            colormap=colormap,
            cbar=True,
            cbar_label="ΔΔ error rate",
            ax=axes,
        )
    else:
        raise ValueError(f"plot_as {plot_as} not recognized")


def plot_egocentric_error_heatmap(
    hm,
    colormap="Reds",
    vmin=None,
    vmax=None,
    cbar=True,
    cbar_label="error rate",
    arrow_color="forestgreen",
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.heatmap(
        hm,
        ax=ax,
        square=True,
        cmap=colormap,
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
        fc=arrow_color,
        ec=arrow_color,
        transform=ax.transAxes,  # CRITICAL
        zorder=5,
    )
    ax.add_patch(arrow)


# %% get egocentric error map df


def get_stats_df(ego_df):
    """ """
    x_min, x_max = ego_df.ego_goal_x.min(), ego_df.ego_goal_x.max()
    y_min, y_max = ego_df.ego_goal_y.min(), ego_df.ego_goal_y.max()
    stats = []
    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            _df = ego_df[(ego_df.ego_goal_x == x) & (ego_df.ego_goal_y == y)]
            stat_df = mixed_anova(
                dv="error_rate",
                within="stim_trial",
                between="condition",
                subject="subject_ID",
                data=_df,
            )
            stat_row = stat_df[stat_df.Source == "Interaction"].iloc[0]
            stats.append(
                {
                    "x": x,
                    "y": y,
                    "F": stat_row["F"],
                    "p": stat_row["p-unc"],
                }
            )
    stats_df = pd.DataFrame(stats)
    # multiple comparisons correction
    reject, pvals_corrected, _, _ = multipletests(stats_df["p"], method="fdr_bh")
    stats_df["p_corrected"] = pvals_corrected
    stats_df["reject_null"] = reject
    return stats_df


def get_egocentric_error_map_df(
    error_df,
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    wmax=5,
    smooth_sigma_levels=(0.0, 1, 2.5),
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
            heatmap = get_egocentric_error_heatmap(sub_df, wmax)
            if smooth_sigma_levels:
                heatmap = radial_smooth_heatmap(
                    heatmap,
                    sigma_levels=smooth_sigma_levels,
                )
            long_hm = heatmap.stack().reset_index(name="error_rate")
            long_hm["subject_ID"] = subject
            long_hm["condition"] = sub_df.condition.unique()[0]
            long_hm["stim_trial"] = stim_trial
            dfs.append(long_hm)
    hm_df = pd.concat(dfs, ignore_index=True)
    return hm_df


def get_egocentric_error_heatmap(df, wmax=5):
    """ """
    # filter for egocentric range
    ego_df = df[df.ego_goal_x.between(-wmax, wmax) & df.ego_goal_y.between(-wmax, wmax)]
    ego_grouped = ego_df.groupby(["ego_goal_x", "ego_goal_y"])
    error_rate = ego_grouped.error.sum() / ego_grouped.error.size()
    hm = error_rate.unstack(0)
    return hm


def radial_smooth_heatmap(df, sigma_levels=(0.0, 1.0, 2.0)):
    arr = np.asarray(df.values, dtype=float)
    H, W = arr.shape

    # ensure sigma_levels is a numpy array (so array indexing works)
    sigma_levels = np.asarray(sigma_levels, dtype=float)
    if sigma_levels.ndim != 1 or sigma_levels.size < 2:
        raise ValueError("sigma_levels must be a 1D sequence with at least two values")

    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max() if r.max() > 0 else r

    smin, smax = sigma_levels[0], sigma_levels[-1]
    sigma_target = smin + (smax - smin) * r_norm
    sigma_target = np.clip(sigma_target, smin, smax)

    blurred = np.stack([gaussian_filter(arr, sigma=s) for s in sigma_levels], axis=0)  # (L, H, W)

    inds = np.searchsorted(sigma_levels, sigma_target, side="right") - 1
    inds = np.clip(inds, 0, len(sigma_levels) - 2)

    # now sigma_levels[inds] works because sigma_levels is a numpy array
    s0 = sigma_levels[inds]
    s1 = sigma_levels[inds + 1]
    alpha = (sigma_target - s0) / (s1 - s0 + 1e-12)

    v0 = blurred[inds, yy, xx]
    v1 = blurred[inds + 1, yy, xx]

    smoothed = (1.0 - alpha) * v0 + alpha * v1
    return pd.DataFrame(smoothed, index=df.index, columns=df.columns)
