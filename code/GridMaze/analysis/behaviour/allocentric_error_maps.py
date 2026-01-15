"""
lost all my code ... :(
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx

from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.behaviour import errors as be

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)


# %%  Functions


def plot_missed_paths_summary(results_df, plot_as="raw", axes=None):
    """ """
    assert plot_as in ["raw", "delta", "delta_delta"]

    return


def get_missed_paths_df(
    error_df,
    maze_name="maze_1",
    e="error",
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    stim_only=True,
):
    """ """
    # filter data
    df = be._filter_error_df(
        error_df,
        e=e,
        stim_day_range=stim_day_range,
        outlier_thres=outlier_thres,
        stim_only=stim_only,
    )
    df = df[df.maze_name == maze_name]

    # load maze stuff
    simple_maze = mr.get_simple_maze(maze_name)
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    label2coord = mr.get_maze_label2coord(simple_maze)
    coord2label = {v: k for k, v in label2coord.items()}
    all_shortest_paths = dict(nx.all_pairs_all_shortest_paths(extended_maze))

    # loop over subjects to generate a map of missed paths (as norm rate)
    dfs = []
    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            sub_df = df[(df.subject_ID == subject) & (df.stim_trial == stim_trial)]
            condition = sub_df.condition.unique()[0]
            mp_all = get_missed_paths_heatmap(sub_df, label2coord, coord2label, all_shortest_paths)
            mp_err = get_missed_paths_heatmap(sub_df[sub_df[e]], label2coord, coord2label, all_shortest_paths)
            mp_norm = mp_err / mp_all.replace(0, np.nan)
            dfs.append(
                pd.DataFrame(
                    {
                        "subject_ID": subject,
                        "condition": condition,
                        "stim_trial": stim_trial,
                        "maze_position": mp_norm.index,
                        "missed_path_rate": mp_norm.values,
                    }
                )
            )
    results_df = pd.concat(dfs, ignore_index=True)
    return results_df


def get_missed_paths_heatmap(df, label2coord, coord2label, all_shortest_paths):
    """ """
    # initialise heatmap
    hm = pd.Series(0, index=label2coord.keys())

    # loop over decisions (df rows) and tally locations on shortest-path to goal
    for _, row in df.iterrows():
        loc = label2coord[row.maze_position]
        goal = label2coord[row.goal]
        path = all_shortest_paths[loc][goal]
        # randomly select one of the possible shortest paths
        path = path[np.random.randint(len(path))]
        for n in path:
            hm.loc[coord2label[n]] += 1

    return hm
