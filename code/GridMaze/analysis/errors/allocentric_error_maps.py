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
from GridMaze.analysis.errors import errors as be
from GridMaze.analysis.processing.get_navigation_dfs import get_cardinal_movement_direction

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Allocentric error maps


def plot_allocentric_error_map_summary(
    results_df,
    maze_name="maze_1",
    plot_as="raw",
    colormap="Reds",
    vmin=None,
    axes=None,
):
    assert plot_as in ["raw", "delta", "delta_delta"]

    # filter for maze
    df = results_df[results_df.maze_name == maze_name]

    # load maze for plotting
    simple_maze = mr.get_simple_maze(maze_name)
    plot_kwargs = {
        "silhouette_edge_size": 6,
        "silhouette_node_size": 150,
        "star_base_length": 0.055,
        "max_point_length": 0.065,
    }

    if plot_as == "raw":
        # average hms across subjects in each group x stim condition
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "maze_position", "optimal_action"]).error_rate.mean()
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
                    **plot_kwargs,
                )

    elif plot_as == "delta":
        # take delta stim_on - stim_off within subject, then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))
        pivot_df = df.pivot_table(
            values="error_rate",
            index=["condition", "subject_ID", "maze_position", "optimal_action"],
            columns="stim_trial",
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        _max = delta_df.groupby(level=[0, 1]).mean().max()
        _min = -_max if vmin is None else vmin
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
                allow_negative=True,
                **plot_kwargs,
            )

    elif plot_as == "delta_delta":
        # take within subject delta then aver across groups and take difference between
        # group averages for delta detla missed path rate
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            values="error_rate",
            index=["condition", "subject_ID", "maze_position", "optimal_action"],
            columns="stim_trial",
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        delta_delta = delta_df.loc["opto"] - delta_df.loc["control"]
        _max = delta_delta.groupby(level=0).mean().max()
        _min = -_max if vmin is None else vmin
        mp.plot_directed_heatmap(
            simple_maze,
            delta_delta,
            ax=axes,
            colormap=colormap,
            value_label="ΔΔ error rate",
            fixed_vmin=_min,
            fixed_vmax=_max,
            allow_negative=True,
            **plot_kwargs,
        )

    return


def get_allocentric_error_maps_df(
    error_df,
    e="error",
    ignore_goal_pass_errors=True,
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
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
    if ignore_goal_pass_errors:
        df = df[~df.goal_pass_error]

    # get state-action error rate map per subject, stim, maze
    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        _df = df[df.maze_name == maze_name]
        for subject in SUBJECT_IDS:
            for stim_trial in [True, False]:
                sub_df = _df[(_df.subject_ID == subject) & (_df.stim_trial == stim_trial)]
                # get error map
                edf = sub_df.groupby(["maze_position", "optimal_action"])[e].mean().reset_index(name="error_rate")
                edf["subject_ID"] = subject
                edf["condition"] = sub_df.condition.unique()[0]
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
    colormap="Reds",
    vmin=None,
    axes=None,
):
    """ """
    assert plot_as in ["raw", "delta", "delta_delta"]

    # filter for maze
    df = results_df[results_df.maze_name == maze_name]

    # load maze for plotting
    simple_maze = mr.get_simple_maze(maze_name)
    plot_kwargs = {
        "silhouette_edge_size": 6,
        "silhouette_node_size": 150,
        "star_base_length": 0.05,
        "max_point_length": 0.03,
    }

    if plot_as == "raw":
        # average hms across subjects in each group x stim condition
        if axes is None:
            f, axes = plt.subplots(2, 2, figsize=(6, 6))
        plot_df = df.groupby(["condition", "stim_trial", "maze_position", "direction"]).missed_path_rate.mean()
        _max = plot_df.groupby(level=[0, 1, 2]).mean().max()
        _min = 0 if vmin is None else vmin
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
                    value_label="miss rate",
                    fixed_vmin=_min,
                    fixed_vmax=_max,
                    **plot_kwargs,
                )

    elif plot_as == "delta":
        # take delta stim_on - stim_off within subject, then average across groups
        if axes is None:
            f, axes = plt.subplots(1, 2, figsize=(6, 3))

        pivot_df = df.pivot_table(
            values="missed_path_rate",
            index=["condition", "subject_ID", "maze_position", "direction"],
            columns="stim_trial",
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        _max = delta_df.groupby(level=[0, 1]).mean().max()
        _min = -_max if vmin is None else vmin
        for ax, group in zip(axes, ["control", "opto"]):
            ax.set_title(f"{group} (stim_on - stim_off)")
            hm = delta_df.loc[group]
            mp.plot_directed_heatmap(
                simple_maze,
                hm,
                ax=ax,
                colormap=colormap,
                value_label="Δ miss rate",
                fixed_vmin=_min,
                fixed_vmax=_max,
                allow_negative=True,
                **plot_kwargs,
            )

    elif plot_as == "delta_delta":
        # take within subject delta then aver across groups and take difference between
        # group averages for delta detla missed path rate
        if axes is None:
            f, axes = plt.subplots(1, 1, figsize=(3, 3))
        pivot_df = df.pivot_table(
            values="missed_path_rate",
            index=["condition", "subject_ID", "maze_position", "direction"],
            columns="stim_trial",
        )
        delta_df = pivot_df.diff(axis=1)[True].groupby(level=[0, 2, 3]).mean()
        delta_delta = delta_df.loc["opto"] - delta_df.loc["control"]
        _max = delta_delta.groupby(level=0).mean().max()
        _min = -_max if vmin is None else vmin
        mp.plot_directed_heatmap(
            simple_maze,
            delta_delta,
            ax=axes,
            colormap=colormap,
            value_label="ΔΔ miss rate",
            fixed_vmin=_min,
            fixed_vmax=_max,
            allow_negative=True,
            **plot_kwargs,
        )


def get_missed_paths_df(
    error_df,
    e="error",
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    ignore_goal_pass_errors=True,
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
    if ignore_goal_pass_errors:
        df = df[~df.goal_pass_error]

    dfs = []
    for maze_name in ["maze_1", "maze_2"]:
        _df = df[df.maze_name == maze_name]
        # load maze stuff
        simple_maze = mr.get_simple_maze(maze_name)
        extended_maze = mr.get_extended_simple_maze(simple_maze)
        all_state_actions = mr.get_maze_place_direction_pairs(simple_maze)
        label2coord = mr.get_maze_label2coord(simple_maze)
        coord2label = {v: k for k, v in label2coord.items()}
        all_shortest_paths = dict(nx.all_pairs_all_shortest_paths(extended_maze))

        # loop over subjects to generate a map of missed paths (as norm rate)

        for subject in SUBJECT_IDS:
            for stim_trial in [True, False]:
                sub_df = _df[(_df.subject_ID == subject) & (_df.stim_trial == stim_trial)]
                condition = sub_df.condition.unique()[0]
                mp_all = get_missed_paths_heatmap(
                    sub_df,
                    all_state_actions,
                    label2coord,
                    coord2label,
                    all_shortest_paths,
                )
                mp_err = get_missed_paths_heatmap(
                    sub_df[sub_df[e]],
                    all_state_actions,
                    label2coord,
                    coord2label,
                    all_shortest_paths,
                )
                mp_norm = mp_err / mp_all.replace(0, np.nan)
                dfs.append(
                    pd.DataFrame(
                        {
                            "subject_ID": subject,
                            "condition": condition,
                            "maze_name": maze_name,
                            "stim_trial": stim_trial,
                            "maze_position": mp_norm.index.get_level_values(0),
                            "direction": mp_norm.index.get_level_values(1),
                            "missed_path_rate": mp_norm.values,
                        }
                    )
                )
    results_df = pd.concat(dfs, ignore_index=True)
    return results_df


def get_missed_paths_heatmap(df, all_state_actions, label2coord, coord2label, all_shortest_paths):
    """ """
    # initialise heatmap
    hm = pd.Series(0, index=pd.MultiIndex.from_tuples(all_state_actions))

    # loop over decisions (df rows) and tally locations on shortest-path to goal
    for _, row in df.iterrows():
        loc = label2coord[row.maze_position]
        goal = label2coord[row.goal]
        path = all_shortest_paths[loc][goal]
        # randomly select one of the possible shortest paths
        path = path[np.random.randint(len(path))]
        states = [coord2label[n] for n in path[:-1]]
        actions = [_get_action(nxt, loc) for loc, nxt in zip(path[:-1], path[1:])]
        state_actions = list(zip(states, actions))
        for sa in state_actions:
            hm.loc[sa] += 1

    return hm


def _get_action(pos, prev_pos):
    """
    Taken from GridMaze.analysis.processing.get_navigation_dfs
    get_cardinal_movement_direction
    """
    position_type = "node" if isinstance(pos[0], int) == True else "edge"
    last_position_type = "node" if isinstance(prev_pos[0], int) == True else "edge"
    if position_type == last_position_type:
        raise ValueError("position and last_position must be different types")
    if position_type == "node" and last_position_type == "edge":
        dx = (pos[0] - prev_pos[0][0]) + (pos[0] - prev_pos[1][0])
        dy = (pos[1] - prev_pos[0][1]) + (pos[1] - prev_pos[1][1])
    elif position_type == "edge" and last_position_type == "node":
        dx = (pos[0][0] - prev_pos[0]) + (pos[1][0] - prev_pos[0])
        dy = (pos[0][1] - prev_pos[1]) + (pos[1][1] - prev_pos[1])
    if dx > 0:
        cardinal_direction = "E"
    elif dx < 0:
        cardinal_direction = "W"
    elif dy > 0:
        cardinal_direction = "N"
    elif dy < 0:
        cardinal_direction = "S"
    return cardinal_direction
