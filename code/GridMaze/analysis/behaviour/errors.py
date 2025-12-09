"""
Script for quantifying navigation errors.
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
from GridMaze.analysis.behaviour import trajectory_plotting as tp
from GridMaze.analysis.core import get_sessions as gs


# %% Global Variables


# %% Functions


def test(sessions=None):
    if sessions is None:
        print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            stim_only=True,
            with_data=["trials_df", "navigation_df", "trajectory_decisions_df"],
            must_have_data=True,
        )
    dfs = []
    for session in sessions:
        print(session.name)
        errors_df = get_nav_errors_df(session)
        dfs.append(errors_df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def get_nav_errors_df(
    session,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 120, "smooth_SD": 4, "min_consecutive": 0.2},
):
    """ """
    # load data
    simple_maze = session.simple_maze()
    trials_df = session.trials_df.copy()
    decisions_df = session.trajectory_decisions_df
    navigation_df = session.navigation_df
    # def constant vars
    trials = trials_df.trial.unique()
    trials_df.set_index("trial", inplace=True)
    decision_points = tp.get_decision_nodes(simple_maze)
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    error_results = []
    # loop over trials & calculate no. of errors (non-optimal actions at decision points)
    for trial in trials:
        stim_trial = bool(trials_df.loc[trial, ("stim_trial", "")])
        trials_since_stim = trials_df.loc[trial, ("trials_since_stim", "")]
        # filter for trial
        nav_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
        dec_df = decisions_df[(decisions_df.trial == trial) & (decisions_df.trial_phase == "navigation")]
        if nav_df.empty or dec_df.empty:
            continue
        goal = nav_df.goal.unique()[0]

        if first_goal_sight:
            first_idx, first_time = tp.get_first_goal_sight(nav_df, **goal_sight_kwargs)
            if first_idx is not None:
                nav_df = nav_df[nav_df.time >= first_time]
                dec_df = dec_df[dec_df.time >= first_time]

        # filter for decision points
        dec_df = dec_df[dec_df.maze_position.isin(decision_points)]

        for idx, row in dec_df.iterrows():
            pos = row.maze_position
            action = row.action
            optimal_actions = tp.get_optimal_action(pos, goal, simple_maze, location_label2coord)
            error = action not in optimal_actions
            error_results.append(
                {
                    "trial": trial,
                    "time": row.time,
                    "error": error,
                    "stim_trial": stim_trial,
                    "stim_on": row.stim_on,
                    "trials_since_stim": trials_since_stim,
                }
            )

    errors_df = pd.DataFrame(error_results)
    # add info for combining across sessions
    errors_df["subject_ID"] = session.subject_ID
    errors_df["condition"] = session.condition
    errors_df["maze_name"] = session.maze_name
    errors_df["day_on_maze"] = session.day_on_maze
    errors_df["stim_session"] = session.stim
    errors_df["stim_day"] = session.stim_day
    return errors_df
