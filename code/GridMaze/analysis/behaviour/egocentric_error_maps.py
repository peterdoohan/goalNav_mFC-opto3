"""
visualize egocentric error maps,
i.e. where was the goal relative to subject POV when they make navigational errors
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from joblib import delayed, Parallel
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrow

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.behaviour import errors as be
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
    ego_cols = ["ego_goal_x", "ego_goal_y"]
    error_counts = ego_df.groupby(ego_cols).error.sum().unstack(0)  # x,y (ego)
    state_visits = ego_df.groupby(ego_cols).size().unstack(0)  # x,y (ego)
    error_rate = error_counts / state_visits
    return error_rate


# %% error df functions


def get_error_df(sessions=None, verbose=False, n_jobs=-1):
    """ """
    if sessions is None:
        if verbose:
            print("Loading all expert stim sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            stim_only=True,
            with_data=["trajectories_df", "trial_info_df", "trials_df"],
            must_have_data=True,
        )

    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(delayed(get_session_error_df)(s) for s in sessions)
    else:
        dfs = []
        for s in sessions:
            if verbose:
                print(f"Processing session: {s.name}")
            df = get_session_error_df(s)
            dfs.append(df)
    error_df = pd.concat(dfs, ignore_index=True)
    return error_df


def get_session_error_df(session):
    """ """
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    simple_maze = session.simple_maze()
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze))
    label2coord = mr.get_maze_label2coord(simple_maze)
    # get node by node transitions df
    error_df = sh.get_decisions_df(
        session,
        navigation_only=True,
        remove_stim_trials=False,
        n_history=None,
    )
    # add stim trial information
    error_df["stim_trial"] = error_df.trial.map(trials_df.stim_trial.to_dict())

    # add steps-to-goal
    error_df["steps_to_goal"] = error_df.apply(
        lambda row: get_steps_to_goal(
            row["maze_position"],
            row["goal"],
            label2coord,
            shortest_path_lengths,
        ),
        axis=1,
    )

    # loop over trials and add errors
    error_masks = []
    time_in_trials = []
    for t in trials_df.index:
        _df = error_df[error_df.trial == t]
        if _df.empty:
            continue
        error_masks.append(get_error_mask(_df.steps_to_goal))
        cue_time = trials_df.loc[t, ("time", "cue")]
        time_in_trials.append(_df.time.sub(cue_time).values)
    error_df["error"] = np.hstack(error_masks)
    error_df["time_in_trial"] = np.hstack(time_in_trials)

    # add egocentric goal coords
    ego_goal_coords = error_df.apply(
        lambda row: get_egocentric_goal_coords(
            row["maze_position"],
            row["action"],
            row["goal"],
            label2coord,
        ),
        axis=1,
    )
    error_df = pd.concat([error_df, ego_goal_coords], axis=1)
    error_df["subject_ID"] = session.subject_ID
    error_df["condition"] = session.condition
    return error_df


def get_egocentric_goal_coords(loc, action, goal, label2coord):
    """ """
    loc_coord = label2coord[loc]
    goal_coord = label2coord[goal]
    if action == "N":
        x = goal_coord[0] - loc_coord[0]
        y = goal_coord[1] - loc_coord[1]
    elif action == "S":
        x = loc_coord[0] - goal_coord[0]
        y = loc_coord[1] - goal_coord[1]
    elif action == "E":
        x = goal_coord[1] - loc_coord[1]
        y = loc_coord[0] - goal_coord[0]
    elif action == "W":
        x = loc_coord[1] - goal_coord[1]
        y = goal_coord[0] - loc_coord[0]
    else:
        raise ValueError("invalid action")
    return pd.Series({"ego_goal_x": int(x), "ego_goal_y": int(y)})


def get_steps_to_goal(loc, goal, label2coord, shortest_path_lengths):
    """ """
    loc_coord = label2coord[loc]
    goal_coord = label2coord[goal]
    return shortest_path_lengths[loc_coord][goal_coord]


def get_error_mask(steps, n=1):
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


# %% sanity check plotting functions


def plot_trial_distance_to_goal(error_df, trial=2, ax=None):
    """ """

    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("steps-to-goal")
    ax.set_title(f"trial {trial}")

    # filter for trial data
    trial_df = error_df[(error_df.trial == trial)]

    # plot steps over time
    steps = trial_df.steps_to_goal.values
    time = trial_df.time_in_trial
    stim_on = trial_df.stim_on.values
    error_mask = trial_df.error

    if np.any(stim_on):
        ax.plot(time[stim_on], steps[stim_on], color="#0077FF", lw=6, alpha=0.5)
    ax.plot(time, steps, color="black", lw=1)
    ax.scatter(time, steps, color="black", s=0.75)
    # plot errors
    if np.any(error_mask):
        etime = time[error_mask]
        y = np.clip(steps[error_mask] - 1.5, a_min=0, a_max=None)
        ax.scatter(etime, y, color="red", marker="x")
