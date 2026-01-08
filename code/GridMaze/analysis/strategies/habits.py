""" """

# %% Imports
from imagecodecs import none_check
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.processing import get_trajectory_decisions_dfs as td

# %% Global Variables


# %% Functions


def test(
    session,
    subject_df,
    stim_day_range=None,
    n_history=2,
):
    """ """
    assert f"history_{n_history}" in subject_df.columns
    # filter subject df
    data_df = subject_df.copy()
    if stim_day_range is not None:
        data_df = subject_df[subject_df.total_stim_days.between(*stim_day_range)]
    data_df = data_df[data_df.maze_name == session.maze_name]
    # exclude current session
    data_df = data_df[data_df.session_name != session.name]
    # keep only relevant columns
    history_cols = [f"history_{i}" for i in range(1, n_history + 1)]
    keep_cols = ["maze_position"] + history_cols + ["action"]
    data_df = data_df[keep_cols]
    # exclude decisions where history is not defined
    data_df = data_df.dropna()

    # get probs
    counts = data_df.groupby(keep_cols).size().reset_index(name="count")
    counts["prob"] = counts["count"] / counts.groupby(history_cols)["count"].transform("sum")
    return counts


def get_subject_decisions_df(
    subject_ID,
    navigation_only=True,
    remove_stim_trials=True,
    n_history=10,
    n_jobs=-1,
):
    sessions = gs.get_maze_sessions(
        subject_IDs=[subject_ID],
        total_stim_days="all",
        with_data=["trajectories_df", "trial_info_df", "trials_df"],
        must_have_data=True,
        verbose=False,
    )
    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(get_decisions_df)(s, navigation_only, remove_stim_trials, n_history) for s in sessions
        )
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.concat(
            [get_decisions_df(s, navigation_only, remove_stim_trials, n_history) for s in sessions],
            ignore_index=True,
        )

    return df


# %%


def get_decisions_df(
    session,
    navigation_only=True,
    remove_stim_trials=True,
    n_history=10,
):
    """
    coppied from analysis.processing.get_trajectory_decision_dfs for convience
    """
    # load relevant processed data
    trials_df = session.trials_df
    simple_maze = session.simple_maze()
    trajectories_df = session.trajectories_df
    frames_trial_info_df = session.trial_info_df
    # get preliminary df (with data frome every frame)
    decisions_df = pd.concat(
        [
            trajectories_df[[("time", ""), ("maze_position", "simple")]],
            frames_trial_info_df,
        ],
        axis=1,
    )
    decisions_df.columns = [c[0] if isinstance(c, tuple) else c for c in decisions_df.columns]
    decisions_df["session_name"] = session.name
    decisions_df["maze_name"] = session.maze_name
    decisions_df["total_stim_days"] = session.total_stim_days
    trial_unique_ID = session.name + "_trial" + decisions_df["trial"].astype(str)
    trial_unique_ID[trial_unique_ID.apply(lambda x: "nan" in x)] = np.nan
    decisions_df["trial_unique_ID"] = trial_unique_ID
    # distill trajectory decisions df to only one frame from each sequental node visit
    decisions_df["maze_position_shifted"] = decisions_df.maze_position.shift(1)
    decisions_df["maze_position_change"] = decisions_df.maze_position != decisions_df.maze_position_shifted
    decisions_df = decisions_df[decisions_df.maze_position_change]
    decisions_df = decisions_df.drop(columns=["maze_position_shifted", "maze_position_change"])
    decisions_df.reset_index(drop=True, inplace=True)
    # correct backtracking by first reducing traj to single nodes transitions and add back edges if required
    node_mask = decisions_df.maze_position.apply(lambda x: len(x.split("-")) == 1)
    decisions_df = decisions_df[node_mask]
    node_trajectory = decisions_df.maze_position
    decisions_df = decisions_df[~(node_trajectory == node_trajectory.shift(1))]
    node_traj = decisions_df.maze_position
    actions = td.get_trajectory_actions(node_traj, simple_maze)
    decisions_df["action"] = actions
    decisions_df = decisions_df[:-1].reset_index(drop=True)  # last action not defined
    # filter further based on input
    if navigation_only:
        decisions_df = decisions_df[decisions_df.trial_phase == "navigation"]
    if remove_stim_trials:
        stim_trials = trials_df[trials_df.stim_trial].trial.unique()
        decisions_df = decisions_df[~decisions_df.trial.isin(stim_trials)]

    # add position histories
    for i in range(1, n_history + 1):
        decisions_df[f"history_{i}"] = decisions_df.groupby("trial_unique_ID")["maze_position"].shift(i)

    return decisions_df
