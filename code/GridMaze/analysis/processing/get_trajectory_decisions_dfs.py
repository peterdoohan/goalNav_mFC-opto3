"""This module creates a dataframe the tracks navigation decisions through the entire trajectory of a session"""

# %% Imports
import random
import numpy as np
import pandas as pd
import networkx as nx
from GridMaze.analysis.core import load_data
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr
from GridMaze.preprocessing import get_maze_trajectories as gmt
from GridMaze.analysis.processing.get_navigation_dfs import get_cardinal_movement_direction

# %% Global variables

# %% Main Functions


def get_trajectory_decisions_df(
    processed_data_path,
    with_edges=True,
    correct_backtracking=True,
    run_trajectory_qc=True,
):
    """
    Creates a dataframe that tracks navigation decisions through the entire trajectory of a session.

    Parameters:
    -----------
    subject_session_path : str
        The path to the session processed data from instead the processed data folder eg, "m2/2021-01-01_12-00-00".
    with_edges : bool, optional
        Whether to include edges between adjacent node visits in the trajectory decisions dataframe. Default is True.
    correct_backtracking : bool, optional
        Whether to correct backtracking in the trajectory decisions dataframe. Default is True.
    run_trajectory_qc : bool, optional
        Whether to run trajectory quality control on the trajectory decisions dataframe. Default is True.

    Returns:
    --------
    trajectory_decisions_df : pandas.DataFrame
        The trajectory decisions dataframe, with columns for subject ID, maze number, day on maze, time, trial, trial phase,
        goal, maze position, and action.
    """
    # load relevant processed data
    session_info = load_data.load(processed_data_path / "session_info.json")
    edges = session_info["maze_structure"]
    simple_maze = mr.simple_maze(edges)
    skeleton_maze = mr.skeleton_maze(edges)
    trajectories_df = load_data.load(processed_data_path / "frames.trajectories.htsv")
    frames_trial_info_df = load_data.load(processed_data_path / "frames.trialInfo.htsv")
    # get preliminary df (with data frome every frame)
    trajectory_decisions_df = pd.concat(
        [
            trajectories_df[[("time", ""), ("maze_position", "simple")]],
            frames_trial_info_df,
        ],
        axis=1,
    )
    trajectory_decisions_df.columns = [c[0] if isinstance(c, tuple) else c for c in trajectory_decisions_df.columns]
    trajectory_decisions_df["subject_ID"] = session_info["subject_ID"]
    trajectory_decisions_df["condition"] = session_info["condition"]
    trajectory_decisions_df["maze_name"] = session_info["maze_name"]
    trajectory_decisions_df["day_on_maze"] = session_info["day_on_maze"]
    trajectory_decisions_df["stim"] = session_info["stim"]
    trajectory_decisions_df["stim_day"] = session_info["stim_day"]
    trial_unique_ID = gs.get_session_name(session_info) + "_trial" + trajectory_decisions_df["trial"].astype(str)
    trial_unique_ID[trial_unique_ID.apply(lambda x: x.split("_")[-1]) == "nan"] = np.nan
    trajectory_decisions_df["trial_unique_ID"] = trial_unique_ID
    # distill trajectory decisions df to only one frame from each sequental node visit
    trajectory_decisions_df["maze_position_shifted"] = trajectory_decisions_df.maze_position.shift(1)
    trajectory_decisions_df["maze_position_change"] = (
        trajectory_decisions_df.maze_position != trajectory_decisions_df.maze_position_shifted
    )
    trajectory_decisions_df = trajectory_decisions_df[trajectory_decisions_df.maze_position_change]
    trajectory_decisions_df = trajectory_decisions_df.drop(columns=["maze_position_shifted", "maze_position_change"])
    trajectory_decisions_df.reset_index(drop=True, inplace=True)
    if not correct_backtracking:
        actions = get_trajectory_actions(trajectory_decisions_df.maze_position, simple_maze)
    else:
        # correct backtracking by first reducing traj to single nodes transitions and add back edges if required
        node_mask = trajectory_decisions_df.maze_position.apply(lambda x: len(x.split("-")) == 1)
        trajectory_decisions_df = trajectory_decisions_df[node_mask]
        node_trajectory = trajectory_decisions_df.maze_position
        trajectory_decisions_df = trajectory_decisions_df[~(node_trajectory == node_trajectory.shift(1))]
        if not with_edges:
            node_traj = trajectory_decisions_df.maze_position
            actions = get_trajectory_actions(node_traj, simple_maze)
        else:
            # artificially add edges between adjacent node visits
            trajectory_decisions_df = add_interpolated_edges_to_trajectory_decisions_df(
                trajectory_decisions_df, simple_maze
            )
            actions = get_node_edges_trajectory_actions(trajectory_decisions_df, simple_maze)
    trajectory_decisions_df["action"] = actions
    # add dadd shortest path distance to goal column
    trajectory_decisions_df["geodesic_distance_to_goal"] = get_path_distances_to_goalf(
        trajectory_decisions_df, simple_maze, skeleton_maze
    )
    trajectory_decisions_df["steps_to_goal"] = get_n_steps_to_goal_df(trajectory_decisions_df, simple_maze)
    if run_trajectory_qc:
        if with_edges:
            assert trajectory_qc(trajectory_decisions_df.maze_position, simple_maze), "trajectory failed QC"
        else:
            assert node_trajectory_qc(trajectory_decisions_df.maze_position, simple_maze), "trajectory failed QC"
    return trajectory_decisions_df[:-1].reset_index(drop=True)  # last action not defined


# %% Real behaviour trajectory df supporting functions


def get_node_edges_trajectory_actions(trajectory_decisions_df, simple_maze):
    traj = trajectory_decisions_df.maze_position
    label2coord = get_maze_label2coord(simple_maze)
    traj = traj.map(label2coord).to_numpy()
    actions = [get_cardinal_movement_direction(traj[i + 1], traj[i]) for i in range(len(traj) - 1)]
    actions.append(np.nan)  # cannot define last action
    return actions


def add_interpolated_edges_to_trajectory_decisions_df(trajectory_decisions_df, simple_maze, type="subject_behaviour"):
    edges_df = trajectory_decisions_df.copy()
    edges = []
    for i in range(len(edges_df) - 1):
        label2coord = get_maze_label2coord(simple_maze)
        coord2label = {v: k for k, v in label2coord.items()}
        node1 = label2coord[edges_df.maze_position.iloc[i]]
        node2 = label2coord[edges_df.maze_position.iloc[i + 1]]
        edge = (node1, node2)
        edges.append(edge)
    edges = gmt.correct_edge_order(edges)
    edges = [coord2label[e] for e in edges]
    edges.append(np.nan)
    edges_df.maze_position = edges
    if type == "subject_behaviour":
        edges_df.time = (edges_df.time + edges_df.time.shift(-1)) / 2
        for col in ["trial", "trial_phase", "goal"]:
            edges_df[col] = edges_df[col].shift(-1)
    elif type == "optimal_behaviour":
        for col in ["trial", "goal"]:
            edges_df[col] = edges_df[col].shift(-1)
    trajectory_decisions_df.reset_index(drop=True, inplace=True)
    edges_df.reset_index(drop=True, inplace=True)
    trajectory_decisions_df = pd.concat([trajectory_decisions_df, edges_df], axis=0).sort_index(kind="merge")[:-1]
    if type == "optimal_behaviour":
        trajectory_decisions_df.reset_index(drop=True, inplace=True)
        trajectory_decisions_df.step = trajectory_decisions_df.index + 1
    return trajectory_decisions_df


def get_trajectory_actions(node_traj, simple_maze):
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    node_traj = node_traj.map(label2coord).to_numpy()
    actions = []
    for i in range(len(node_traj) - 1):
        current_position = node_traj[i]
        next_position = node_traj[i + 1]
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        if dx == 0:
            if dy == 1:
                action = "N"
            elif dy == -1:
                action = "S"
        else:
            if dx == 1:
                action = "E"
            elif dx == -1:
                action = "W"
        actions.append(action)
    actions.append(np.nan)
    return np.array(actions)


def node_trajectory_qc(node_traj, simple_maze):
    """ """
    label2coord = get_maze_label2coord(simple_maze)
    nodes = node_traj.map(label2coord).to_numpy()
    for i in range(len(nodes) - 1):
        if nodes[i + 1] not in simple_maze[nodes[i]]:
            return False
    return True


def trajectory_qc(traj, simple_maze):
    """ """
    label2coord = get_maze_label2coord(simple_maze)
    traj_coords = traj.map(label2coord)
    return gmt.trajectory_qc(traj_coords, simple_maze)


def get_maze_label2coord(simple_maze):
    label2coord = {
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
    }
    return label2coord


# %% Artifical trajectories supporting function


def sample_without_replacement(original_list):
    """
    Generator that yields a random sample from the given list without replacement, infinitely.
    Example:
        >>> lst = [1, 2, 3, 4, 5]
        >>> gen = sample_without_replacement(lst)
        >>> next(gen)
        3
    """
    list_to_sample = original_list.copy()
    while True:
        if not list_to_sample:
            list_to_sample = original_list.copy()
        sample = random.sample(list_to_sample, 1)[0]
        list_to_sample.remove(sample)
        yield sample


# %% Distance/steps to goal calculation functions


def get_path_distances_to_goalf(trajectory_decisions_df, simple_maze, skeleton_maze):
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    skeleton_path_distances = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    label2coord = {
        v: k + (0,) for k, v in nx.get_node_attributes(extended_simple_maze, "label").items()
    }  # add (0,) to make compatible with skeleton maze labels

    def get_path_distance(node1, node2, trial_phase):
        if trial_phase != "navigation":
            return np.nan
        else:
            return skeleton_path_distances[node1][node2]

    distances = pd.concat(
        [  # get maze position and goal as coordinates
            trajectory_decisions_df.maze_position.map(label2coord),
            trajectory_decisions_df.goal.map(label2coord),
            trajectory_decisions_df.trial_phase,
        ],
        axis=1,  # then find the distance betwen them (row by row)
    ).apply(
        lambda row: get_path_distance(row.maze_position, row.goal, row.trial_phase),
        axis=1,
    )
    return distances


def get_n_steps_to_goal_df(trajectory_decisions_df, simple_maze):
    """Not currently using this anymore, replaced with actual distance calculations"""
    extended_simple_maze = mr.get_extended_simple_maze(simple_maze)
    path_distances = dict(nx.shortest_path_length(extended_simple_maze))
    label2coord = {v: k for k, v in nx.get_node_attributes(extended_simple_maze, "label").items()}

    def get_step_distance(node1, node2, trial_phase):
        if trial_phase != "navigation":
            return np.nan
        else:
            return path_distances[node1][node2]

    distances = pd.concat(
        [  # get maze position and goal as coordinates
            trajectory_decisions_df.maze_position.map(label2coord),
            trajectory_decisions_df.goal.map(label2coord),
            trajectory_decisions_df.trial_phase,
        ],
        axis=1,  # then find the distance betwen them (row by row)
    ).apply(
        lambda row: get_step_distance(row.maze_position, row.goal, row.trial_phase),
        axis=1,
    )
    return distances
