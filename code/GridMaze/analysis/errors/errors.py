"""
Quant of errors during navigation
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn import neighbors

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.strategies import habits as sh


# %% Global Variables

MAX_STIM_DURATION = 30

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Group x Stim quantification of errors


def plot_group_by_stim_errors(
    error_df,
    e="error",
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    stim_only=True,
    ax=None,
):
    """
    outlier thres testing:
        "error": 30,
        "repeat_error": 10,
        "goal_pass_error": 10,
    """
    df = _filter_error_df(
        error_df,
        e=e,
        stim_day_range=stim_day_range,
        outlier_thres=outlier_thres,
        stim_only=stim_only,
    )

    # get group x stim
    grouped_df = df.groupby(["condition", "subject_ID", "stim_trial"])
    e_count = grouped_df[e].sum()
    n_trials = grouped_df.trial_unique_ID.size()
    e_rate = e_count / n_trials
    _name = e + "_rate"
    erate_df = e_rate.reset_index(name=_name)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    cp.plot_group_by_stim(
        erate_df,
        y=_name,
        ax=ax,
        print_stats=True,
        legend=False,
    )


def _filter_error_df(
    error_df,
    e="error",
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    stim_only=True,
):
    """ """
    df = error_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        # match time in trial data across stim ON/OFF
        df = df[df.time_in_trial <= MAX_STIM_DURATION]
    if outlier_thres is not None:
        # if there are some massive outliers (eg, due to off task activty)
        trial_counts = error_df.groupby(["trial_unique_ID"])[e].sum()
        outlier_trials = trial_counts[trial_counts > outlier_thres].index
        df = df[~df.trial_unique_ID.isin(outlier_trials)]

    return df


# %% eogcentric error map functions


def get_error_df(sessions=None, verbose=False, n_jobs=-1, save=False):
    """ """
    save_path = RESULTS_PATH / "errors" / "error_df.parquet"
    if not save and save_path.exists():
        if verbose:
            print(f"Loading existing error df from {save_path}")
        error_df = pd.read_parquet(save_path)
        return error_df
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
    error_df = pd.concat(dfs, axis=0)
    error_df.reset_index(drop=True, inplace=True)
    if save:
        if verbose:
            print(f"Saving error df to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_parquet(save_path)
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

    dfs = []
    for t in trials_df.index:
        _df = error_df[error_df.trial == t].copy()
        if _df.empty:
            continue
        # mark errors
        _df["error"] = get_error_mask(_df.steps_to_goal)
        # mark repeat errors
        repeat_errors = _df[_df.error].duplicated(subset=["maze_position", "action"])
        _df["repeat_error"] = repeat_errors.reindex(_df.index, fill_value=False)
        # get goal-pass errors
        _df["goal_pass_error"] = (_df.maze_position == _df.goal) & (_df.index != _df.index[-1])
        # get time in trial
        cue_time = trials_df.loc[t, ("time", "cue")]
        _df["time_in_trial"] = _df.time.sub(cue_time).values
        # get optimal actions
        _df["optimal_action"] = _df.apply(
            lambda row: get_optimal_action(
                row["maze_position"],
                row["goal"],
                shortest_path_lengths,
                simple_maze,
                label2coord,
            ),
            axis=1,
        )
        dfs.append(_df)
    error_df = pd.concat(dfs, ignore_index=True)

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


def get_optimal_action(pos, goal, shortest_path_lengths, simple_maze, label2coord):
    """ """
    pos_coord = label2coord[pos]
    goal_coord = label2coord[goal]
    neighbors = np.array([tuple(n) for n in simple_maze.neighbors(pos_coord)])
    path_lens = np.array([shortest_path_lengths[tuple(neighbor)][goal_coord] for neighbor in neighbors])
    optimal_neighbors = neighbors[path_lens == path_lens.min()]
    if len(optimal_neighbors) == 1:
        optimal_neighbor = optimal_neighbors[0]
    else:
        # randomly choose one of the optimal neighbors
        optimal_neighbor = optimal_neighbors[np.random.choice(len(optimal_neighbors))]
    optimal_action = sh.get_action_from_positions(pos_coord, optimal_neighbor)
    return optimal_action


def get_egocentric_goal_coords(loc, action, goal, label2coord):
    """
    Returns egocentric goal coordinates.
    ego_goal_x: forward (positive ahead)
    ego_goal_y: right (positive to the agent's right)

    """
    xl, yl = label2coord[loc]
    xg, yg = label2coord[goal]

    if action == "N":
        x = xg - xl
        y = yl - yg
    elif action == "S":
        x = xl - xg
        y = yg - yl

    elif action == "E":
        x = yg - yl
        y = xg - xl

    elif action == "W":
        x = yl - yg
        y = xl - xg

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
