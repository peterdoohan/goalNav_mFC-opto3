"""
lost all my code ... :(
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.behaviour import errors as be

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Allocentric error maps


def plot_allocentric_error_map_summary(
    results_df,
    maze_name="maze_1",
    plot_as="raw",
    colormap="viridis",
    vmin=None,
    axes=None,
):
    assert plot_as in ["raw", "delta", "delta_delta"]

    # filter for maze
    df = results_df[results_df.maze_name == maze_name]

    # load maze for plotting
    simple_maze = mr.get_simple_maze(maze_name)

    if plot_as == "raw":
        # average hms across subjects in each group x stim condition
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "maze_position", "action"]).error_rate.mean()
        _max = plot_df.groupby(level=[0, 1, 2]).mean().max()
        for i, group in enumerate(["control", "opto"]):
            for j, stim_trial in enumerate([False, True]):
                ax = axes[i, j]
                ax.set_title(f"{group} - stim:{stim_trial}")
                hm = plot_df.loc[(group, stim_trial)]
                mp.plot_directed_heatmap(
                    simple_maze,
                    hm,
                    ax=ax,
                    colormap=colormap,
                    value_label="error rate",
                    fixed_vmin=0,
                    fixed_vmax=_max,
                    silhouette_edge_size=6,
                    silhouette_node_size=150,
                )

    elif plot_as == "delta":
        # take delta stim_on - stim_off within subject, then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))
        pivot_df = df.pivot_table(
            values="error_rate", index=["condition", "subject_ID", "maze_position", "action"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        if vmin is None:
            _min = delta_df.groupby(level=[0, 1]).mean().min()
        _max = delta_df.groupby(level=[0, 1]).mean().max()

        for ax, group in zip(axes, ["control", "opto"]):
            ax.set_title(f"{group} (stim_on - stim_off)")
            hm = delta_df.loc[group]
            mp.plot_directed_heatmap(
                simple_maze,
                hm,
                ax=ax,
                colormap=colormap,
                value_label="Δ error rate",
                fixed_vmin=_min,
                fixed_vmax=_max,
                silhouette_edge_size=6,
                silhouette_node_size=150,
                allow_negative=True,
            )

    elif plot_as == "delta_delta":
        # take within subject delta then aver across groups and take difference between
        # group averages for delta detla missed path rate
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            values="error_rate", index=["condition", "subject_ID", "maze_position", "action"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        delta_delta = delta_df.loc["opto"] - delta_df.loc["control"]
        if vmin is None:
            _min = delta_delta.min()
        _max = delta_delta.groupby(level=0).mean().max()
        mp.plot_directed_heatmap(
            simple_maze,
            delta_delta,
            ax=axes,
            colormap=colormap,
            value_label="ΔΔ error rate",
            fixed_vmin=0,
            fixed_vmax=_max,
            silhouette_edge_size=6,
            silhouette_node_size=150,
            allow_negative=True,
        )

    return


def get_error_maps_df(
    error_df,
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

    # get state-action error rate map per subject, stim, maze
    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        _df = df[df.maze_name == maze_name]
        for subject in SUBJECT_IDS:
            for stim_trial in [True, False]:
                sub_df = _df[(_df.subject_ID == subject) & (_df.stim_trial == stim_trial)]
                condition = sub_df.condition.unique()[0]
                # get error map
                edf = sub_df.groupby(["maze_position", "action"])[e].mean().reset_index(name="error_rate")
                edf["subject_ID"] = subject
                edf["condition"] = condition
                edf["maze_name"] = maze_name
                edf["stim_trial"] = stim_trial
                dfs.append(edf)

    results_df = pd.concat(dfs, ignore_index=True)
    return results_df


# %%  Missed PathsFunctions


def plot_missed_paths_summary(
    results_df,
    maze_name="maze_1",
    plot_as="raw",
    colormap="viridis",
    vmin=None,
    axes=None,
):
    """ """
    assert plot_as in ["raw", "delta", "delta_delta"]

    # filter for maze
    df = results_df[results_df.maze_name == maze_name]

    # load maze for plotting
    simple_maze = mr.get_simple_maze(maze_name)

    if plot_as == "raw":
        # average hms across subjects in each group x stim condition
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "maze_position"]).missed_path_rate.mean()
        _max = plot_df.max()
        for i, group in enumerate(["control", "opto"]):
            for j, stim_trial in enumerate([False, True]):
                ax = axes[i, j]
                ax.set_title(f"{group} - stim:{stim_trial}")
                hm = plot_df.loc[(group, stim_trial)]
                clabel = "missed path rate" if i == 1 and j == 1 else None
                mp.plot_simple_heatmap(
                    simple_maze,
                    hm,
                    ax=ax,
                    colormap=colormap,
                    value_label=clabel,
                    vmin=0,
                    vmax=_max,
                    edge_size=6,
                    node_size=150,
                )

    elif plot_as == "delta":
        # take delta stim_on - stim_off within subject, then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))
        pivot_df = df.pivot_table(
            values="missed_path_rate", index=["condition", "subject_ID", "maze_position"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2]).mean()
        if vmin is None:
            _min = delta_df.min()
        _max = delta_df.max()
        for ax, group in zip(axes, ["control", "opto"]):
            ax.set_title(f"{group} (stim_on - stim_off)")
            clabel = "Δ missed path rate" if group == "opto" else None
            hm = delta_df.loc[group]
            mp.plot_simple_heatmap(
                simple_maze,
                hm,
                ax=ax,
                colormap=colormap,
                value_label=clabel,
                vmin=_min,
                vmax=_max,
                edge_size=6,
                node_size=150,
                allow_negative=True,
            )

    elif plot_as == "delta_delta":
        # take within subject delta then aver across groups and take difference between
        # group averages for delta detla missed path rate
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            values="missed_path_rate", index=["condition", "subject_ID", "maze_position"], columns="stim_trial"
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2]).mean()
        delta_delta = delta_df.loc["opto"] - delta_df.loc["control"]
        if vmin is None:
            _min = delta_delta.min()
        _max = delta_delta.max()
        mp.plot_simple_heatmap(
            simple_maze,
            delta_delta,
            ax=axes,
            colormap=colormap,
            value_label="ΔΔ missed path rate",
            vmin=0,
            vmax=_max,
            edge_size=6,
            node_size=150,
            allow_negative=True,
        )


def get_missed_paths_df(
    error_df,
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

    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        _df = df[df.maze_name == maze_name]
        # load maze stuff
        simple_maze = mr.get_simple_maze(maze_name)
        extended_maze = mr.get_extended_simple_maze(simple_maze)
        label2coord = mr.get_maze_label2coord(simple_maze)
        coord2label = {v: k for k, v in label2coord.items()}
        all_shortest_paths = dict(nx.all_pairs_all_shortest_paths(extended_maze))

        # loop over subjects to generate a map of missed paths (as norm rate)

        for subject in SUBJECT_IDS:
            for stim_trial in [True, False]:
                sub_df = _df[(_df.subject_ID == subject) & (_df.stim_trial == stim_trial)]
                condition = sub_df.condition.unique()[0]
                mp_all = get_missed_paths_heatmap(sub_df, label2coord, coord2label, all_shortest_paths)
                mp_err = get_missed_paths_heatmap(sub_df[sub_df[e]], label2coord, coord2label, all_shortest_paths)
                mp_norm = mp_err / mp_all.replace(0, np.nan)
                dfs.append(
                    pd.DataFrame(
                        {
                            "subject_ID": subject,
                            "condition": condition,
                            "maze_name": maze_name,
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
