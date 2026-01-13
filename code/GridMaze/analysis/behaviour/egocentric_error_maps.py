"""
visualize egocentric error maps,
i.e. where was the goal relative to subject POV when they make navigational errors
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.behaviour import errors as be
from GridMaze.analysis.strategies import habits as sh

# %% Global Variables

# %% Function


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
