"""
Analyses looking at rate of change of distance to goal aligned to cue, with and without stim across groups
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr


# %% Global Variables
FRAME_RATE = 60

# %% Functions


def get_session_delta_distance_to_goal_df(session, cue_window=5):
    """ """
    window_frames = cue_window * FRAME_RATE

    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    navigation_df = session.navigation_df
    skeleton_maze = session.skeleton_maze()
    label2coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    dD_dts = []
    for trial in trials_df.index.to_list():
        goal_coord = label2coord[trials_df.loc[trial, ("goal", "")] + "_C"]
        cue_indx = navigation_df[navigation_df.trial == trial].index[0]
        locs = navigation_df.iloc[
            cue_indx - window_frames : cue_indx + window_frames + 1
        ].maze_position.skeleton.to_numpy()
        if len(locs) == 0:
            continue  # window out of session bounds
        coords = [label2coord[loc] for loc in locs]
        aligned_dtg = np.array([shortest_path_lengths[c][goal_coord] for c in coords])
        dD_dt = np.diff(aligned_dtg) * FRAME_RATE  # m/s
        dD_dts.append(dD_dt)
    D = np.vstack(dD_dts)  # [trials, timepoints]
    # warp into dataframe for convenience
    aligned_times = np.arange(-cue_window, cue_window + 1 / FRAME_RATE, 1 / FRAME_RATE)[:-1]
    ddtg_df = pd.DataFrame(
        D,
        columns=pd.MultiIndex.from_product([["delta_distance_to_goal"], aligned_times.tolist()]),
    )
    info_df = trials_df.reset_index()[[("trial", ""), ("stim_trial", "")]]
    info_df[("subject_ID", "")] = session.subject_ID
    info_df[("condition", "")] = session.condition
    info_df[("maze_name", "")] = session.maze_name
    info_df[("day_on_maze", "")] = session.day_on_maze
    info_df[("stim_day", "")] = session.stim_day
    info_df[("total_stim_days", "")] = session.total_stim_days
    return pd.concat([info_df, ddtg_df], axis=1)
