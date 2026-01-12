"""
Quant of errors during navigation
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pyparsing import C
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import plotting as mp
from GridMaze.maze import representations as mr

# %% Global Variables

MAX_STIM_DURATION = 30

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% eogcentric error map functions


def test(error_df):
    """ """
    df = _filter_error_data(error_df, maze_name="maze_2", stim_day_range=(4, gs.TOTAL_STIM_DAYS), goal=None)
    return


# %% allocentric error map functions


def plot_delta_delta_allocentric_error_map(
    error_df,
    maze_name="maze_1",
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    goal=None,
    ax=None,
):
    """
    note looking at errors allocentrically divides up the data to the point where
    we are very limited, will be more productive to look at error maps defined egocentrically
    where we can collapose data across goals and mazes more easily
    """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # filter data
    df = _filter_error_data(error_df, maze_name=maze_name, stim_day_range=stim_day_range, goal=goal)
    # further filter by time < max stim duration (30s) in both conditions
    df = df[df.time_in_trial <= MAX_STIM_DURATION]
    trial_type_counts = (
        df[["trial_unique_ID", "condition", "stim_trial"]]
        .drop_duplicates()
        .droplevel(1, axis=1)
        .groupby(["condition", "stim_trial"])
        .count()
    ).trial_unique_ID
    error_counts = (
        df.groupby([("condition", ""), ("stim_on", ""), ("maze_position", "simple")]).size().unstack(level=[0, 1])
    ).dropna()  # remove locations with no errors in some conditions
    # normalise to errors per trial
    for cond in ["control", "opto"]:
        for stim_trial in [True, False]:
            error_counts[(cond, stim_trial)] = error_counts[(cond, stim_trial)].div(
                trial_type_counts.loc[(cond, stim_trial)]
            )
    # compute delta errors across stim within condition
    # then deleta delta errors across conditions
    delta_delta = error_counts.stack(level=0, future_stack=True).diff(axis=1)[True].unstack().diff(axis=1)["opto"]
    # account for missing locs (not enough errors or edge)
    simple_maze = mr.get_simple_maze(maze_name)
    all_locs = mr.get_maze_locations(simple_maze)
    missing_locs = list(set(all_locs) - set(delta_delta.index.to_list()))
    dd_plot = pd.concat([delta_delta, pd.Series(0, index=missing_locs)], axis=0).sort_index()
    # plot simple heatmap
    _max = dd_plot.abs().max()
    mp.plot_simple_heatmap(
        simple_maze,
        dd_plot,
        ax=ax,
        vmin=-_max,
        vmax=_max,
        colormap="coolwarm",
        value_label="ΔΔ errors",
        allow_negative=True,
    )


def plot_delta_delta_error_map(
    error_df,
    maze_name="maze_1",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    goal=None,
    angle_color="black",
    normalise=True,
    min_count=10,
    ax=None,
):
    return


def plot_raw_error_map(
    error_df,
    maze_name="maze_1",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    goal=None,
    angle_color="black",
    normalise=True,
    min_count=10,
    ax=None,
):
    """ """
    simple_maze = mr.get_simple_maze(maze_name)
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # filter data
    df = _filter_error_data(error_df, maze_name, stim_day_range, goal)
    # plot error count heatmap
    _error_counts = df.groupby(("maze_position", "simple")).size()
    all_locs = mr.get_maze_locations(simple_maze)
    missing_locs = list(set(all_locs) - set(_error_counts.index.to_list()))
    error_counts = pd.concat([_error_counts, pd.Series(0, index=missing_locs)], axis=0).sort_index()
    if normalise:
        error_counts = error_counts.div(error_counts.sum())
    mp.plot_simple_heatmap(simple_maze, error_counts, ax=ax, colormap="silver2red", value_label="error count")
    # plot error direction markers
    angle_summary = df.groupby(("maze_position", "simple")).apply(circular_summary, include_groups=True)
    label2pos = mr.get_maze_label2position(mr.get_extended_simple_maze(simple_maze))
    for label, pos in label2pos.items():
        x, y = pos
        if label in angle_summary.index:
            if _error_counts.loc[label] < min_count:
                plot_empty(x, y, ax, c="lightgrey")
                continue  # skip low count locations
            arrow_info = angle_summary.loc[label]
            mean_dir = arrow_info["mean_direction"]
            res_length = arrow_info["resultant_length"]
            plot_error_marker(x, y, np.deg2rad(mean_dir), ax, length=0.2 * res_length, color=angle_color)
        else:
            plot_empty(x, y, ax, c="lightgrey")


def circular_summary(group):
    angles = np.deg2rad(group["head_direction"])

    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))

    mean_angle = np.arctan2(S, C)
    resultant_length = np.sqrt(C**2 + S**2)

    return pd.Series({"mean_direction": np.rad2deg(mean_angle) % 360, "resultant_length": resultant_length})


def plot_error_marker(x, y, theta, ax, color="m", length=0.1, head_length=5, head_width=2, alpha=0.5):
    """add arrow to ax at (x,y) with angle theta (radians)"""
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow = FancyArrowPatch(
        (x, y),
        (x + dx, y + dy),
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        linewidth=2,
        color=color,
        alpha=alpha,
        mutation_scale=1.0,
        zorder=11,
    )
    ax.add_patch(arrow)


def plot_empty(x, y, ax, c="m"):
    """plot small dot at pos: x,y"""
    circle = Circle((x, y), 0.01, color=c, zorder=10)
    ax.add_patch(circle)


def _filter_error_data(error_df, maze_name=None, stim_day_range=(6, gs.TOTAL_STIM_DAYS), goal=None):
    """ """
    df = error_df.copy()
    if maze_name is not None:
        df = df[df.maze_name == maze_name]
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if goal is not None:
        df = df[df.goal == goal]
    return df


# %%


def get_error_df(sessions=None, n=1, verbose=False, n_jobs=-1):
    """ """
    if sessions is None:
        if verbose:
            print("Loading all expert stim sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            stim_only=True,
            with_data=["navigation_df", "trials_df", "trajectory_decisions_df"],
            must_have_data=True,
        )

    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(delayed(get_session_error_df)(s, n) for s in sessions)
    else:
        dfs = []
        for s in sessions:
            if verbose:
                print(f"Processing session: {s.name}")
            df = get_session_error_df(s, n)
            dfs.append(df)
    error_df = pd.concat(dfs, ignore_index=True)
    return error_df


def get_session_error_df(
    session,
    n=1,
):
    """ """
    # load data
    trials_df = session.trials_df.copy()
    navigation_df = session.navigation_df
    trajectory_decisions_df = session.trajectory_decisions_df

    # loop over trials & collect info at error times
    trials_df.set_index("trial", inplace=True)
    results = []
    for trial in trials_df.index:
        # get error times using traj_decisions_df
        td_df = trajectory_decisions_df[
            (trajectory_decisions_df.trial == trial) & (trajectory_decisions_df.trial_phase == "navigation")
        ].copy()
        if td_df.shape[0] <= 2:
            continue  # skip very short trials
        error_mask = get_error_mask(td_df.steps_to_goal, n=n)
        if not np.any(error_mask):
            continue
        error_times = td_df.time.values[error_mask]
        cue_time = trials_df.loc[trial, ("time", "cue")]
        times_in_trial = error_times - cue_time  # align to trial start
        # use navigation df to get higher-res info about loc, etc. at error times
        nav_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")].copy()
        error_df = nav_df.loc[np.array([nav_df.time.sub(et).abs().idxmin() for et in error_times])]
        _df = error_df[
            [
                ("subject_ID", ""),
                ("maze_name", ""),
                ("trial_unique_ID", ""),
                ("trial", ""),
                ("time", ""),
                ("goal", ""),
                ("stim_on", ""),
                ("centroid_position", "x"),
                ("centroid_position", "y"),
                ("maze_position", "simple"),
                ("cardinal_movement_direction", ""),
                ("distance_to_goal", "euclidean"),
                ("distance_to_goal", "geodesic"),
                ("head_direction", "value"),
                ("angle_to_goal", "allocentric"),
                ("angle_to_goal", "egocentric"),
            ]
        ].copy()
        _df[("time_in_trial", "")] = times_in_trial
        results.append(_df)
    error_df = pd.concat(results, axis=0).reset_index(drop=True)
    error_df[("stim_trial")] = error_df.trial.map(trials_df.stim_trial.to_dict())
    error_df[("total_stim_days", "")] = session.total_stim_days
    error_df[("condition", "")] = session.condition
    return error_df


# %%


def plot_trial_distance_to_goal(session, trial=2, ax=None):
    """ """
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    cue_time = trials_df.loc[trial, ("time", "cue")]
    # filter for trial data
    df = session.trajectory_decisions_df.copy()
    trial_df = df[(df.trial == trial) & (df.trial_phase == "navigation")]

    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("steps-to-goal")
    ax.set_title(f"trial {trial}")

    # plot steps over time
    steps = trial_df.steps_to_goal.values
    time = trial_df.time.values
    time = time - cue_time  # align to trial start
    stim_on = trial_df.stim_on.values
    if np.any(stim_on):
        ax.plot(time[stim_on], steps[stim_on], color="#0077FF", lw=6, alpha=0.5)
    ax.plot(time, steps, color="black", lw=1)
    ax.scatter(time, steps, color="black", s=0.75)
    # plot errors
    error_mask = get_error_mask(trial_df.steps_to_goal)
    if np.any(error_mask):
        etime = time[error_mask]
        y = np.clip(steps[error_mask] - 1.5, a_min=0, a_max=None)
        ax.scatter(etime, y, color="red", marker="x")

    return


def get_error_times(session, trial, rel=True):
    """
    convience function
    """
    # filter for trial data
    df = session.trajectory_decisions_df.copy()
    trial_df = df[(df.trial == trial) & (df.trial_phase == "navigation")]
    error_mask = get_error_mask(trial_df.steps_to_goal)
    if error_mask.sum() == 0:
        return None
    error_times = trial_df.time.values[error_mask]
    if rel:
        # align times to trial start
        trials_df = session.trials_df.copy()
        trials_df.set_index("trial", inplace=True)
        cue_time = trials_df.loc[trial, ("time", "cue")]
        error_times = error_times - cue_time
    return error_times


def get_error_mask(steps, n=2):
    """
    True only at the first increase after n consecutive decreases.
    """
    diffs = steps.diff()
    increase = diffs > 0  #
    # previous n diffs were all negative
    prev_n_decreasing = diffs.shift(1).rolling(n, min_periods=n).max() < 0
    # detect is True at the first increasing index i; shift it back to i-1
    error_mask = (increase & prev_n_decreasing).fillna(False)
    error_mask = error_mask.shift(-1)
    error_mask.iloc[-1] = False  # avoid NaN at end
    return error_mask.values.astype(bool)
