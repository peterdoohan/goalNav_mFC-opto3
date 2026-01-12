"""
lib for plotting trial tajectoies for QC and for vis effect of opto
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.processing.get_navigation_strategies_dfs import get_neighbor_cdir
from GridMaze.analysis.behaviour import errors as be

from matplotlib.patches import Wedge, FancyArrowPatch

# %% Global Variables

FRMAE_RATE = 60

# %%


def plot_fancy_trial_trajectory(
    session,
    trial=1,
    smooth_SD=10,
    t_range=None,
    plot_errors=False,
    ax=None,
):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))
    # load data
    navigation_df = session.navigation_df
    simple_maze = session.simple_maze()
    # filter data for specified trial
    df = navigation_df[navigation_df.trial == trial]
    df = df[df.trial_phase == "navigation"]
    # extract trajectory
    x_traj = df.centroid_position.x.values
    y_traj = df.centroid_position.y.values
    stim_trial = df.stim_on.any()
    stim_mask = df.stim_on.values if stim_trial else np.zeros(len(df)).astype(bool)
    time = df.time.values
    time = time - time.min()
    if t_range is not None:
        t_mask = (time >= t_range[0]) & (time <= t_range[1])
    else:
        t_mask = np.ones(len(time)).astype(bool)
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, smooth_SD)
        y_traj = gaussian_filter1d(y_traj, smooth_SD)
    goal = df.goal.unique()[0]
    color2loc = {}
    color2loc[goal] = "gold"
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax=ax,
        color="silver",
        special_location2color=color2loc,
        node_size=175,
        edge_size=6.5,
    )
    if stim_mask.any():
        ax.plot(
            x_traj[t_mask][stim_mask[t_mask]],
            y_traj[t_mask][stim_mask[t_mask]],
            color="#0077FF",
            linewidth=8,
            alpha=0.4,
        )

    # Create line segments from x, y
    points = np.column_stack([x_traj, y_traj]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = segments[t_mask[:-1]]

    # Normalize time for colormap
    norm = Normalize(time.min(), time.max())

    # Create LineCollection
    lc = LineCollection(segments, cmap="Reds", norm=norm)
    lc.set_array(time[t_mask][:-1])  # one value per segment
    lc.set_linewidth(3)
    lc.set_antialiased(True)
    lc.set_capstyle("round")
    lc.set_joinstyle("bevel")

    # Plot
    ax.add_collection(lc)
    if t_range is not None:
        ax.set_title(f"Trial {trial} (t={t_range[0]}-{t_range[1]}s)", fontsize=8)
    else:
        ax.set_title(f"Trial {trial}", fontsize=8)

    if plot_errors:
        error_times = be.get_error_times(session, trial=trial, rel=False)
        if t_range:
            error_times = [
                et for et in error_times if (et - df.time.min() >= t_range[0]) and (et - df.time.min() <= t_range[1])
            ]
        if error_times is not None:
            error_df = df.loc[np.array([(df.time - et).abs().idxmin() for et in error_times])]
            ax.scatter(
                error_df.centroid_position.x.values,
                error_df.centroid_position.y.values,
                marker="x",
                s=100,
                color="m",
                alpha=0.8,
                zorder=10,
            )
    return


# %%


def plot_trial_trajectory(
    session,
    trial=1,
    smooth_SD=5,
    plot_first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 150, "smooth_SD": 5, "min_consecutive": 0.4},
    plot_error_points=False,
    plot_shortest_path=True,
    ax=None,
):
    """ """
    # set up plot
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 5))
    # load data
    navigation_df = session.navigation_df
    simple_maze = session.simple_maze()
    # filter data for specified trial
    df = navigation_df[navigation_df.trial == trial]
    df = df[df.trial_phase == "navigation"]
    # extract trajectory
    x_traj = df.centroid_position.x.values
    y_traj = df.centroid_position.y.values
    if smooth_SD:
        x_traj = gaussian_filter1d(x_traj, smooth_SD)
        y_traj = gaussian_filter1d(y_traj, smooth_SD)
    # define stim
    stim_trial = df.stim_on.any()
    stim_mask = df.stim_on.values if stim_trial else np.zeros(len(df)).astype(bool)
    goal = df.goal.unique()[0]
    # plot
    color2loc = {}
    if plot_shortest_path:
        excess_steps, shortest_path = get_excess_steps(
            session,
            trial,
            return_shortest_path=True,
            first_goal_sight=plot_first_goal_sight,
            goal_sight_kwargs=goal_sight_kwargs,
        )
        color2loc = {sp_loc: "slategrey" for sp_loc in shortest_path}
        ax.text(0.05, 1.35, f"Excess Steps: {excess_steps}", fontsize=12)
    color2loc[goal] = "gold"
    mp.plot_simple_maze_silhouette(
        simple_maze,
        ax=ax,
        color="silver",
        special_location2color=color2loc,
    )
    if stim_mask.any():
        ax.plot(
            x_traj[stim_mask],
            y_traj[stim_mask],
            color="blue",
            linewidth=8,
            alpha=0.4,
        )
    ax.plot(
        x_traj,
        y_traj,
        color="black",
        linewidth=4,
        alpha=0.7,
    )
    if plot_first_goal_sight:
        first_idx, first_time = get_first_goal_sight(df, **goal_sight_kwargs)
        if first_idx is not None:
            nav_frame = df.iloc[first_idx]
            pos = (nav_frame.centroid_position.x, nav_frame.centroid_position.y)
            head_deg = nav_frame.head_direction.values[0]
            plot_fov(ax, pos, head_deg, alpha_deg=goal_sight_kwargs["alpha_deg"])
    if plot_error_points:
        error_times = get_error_times(session, trial=trial)
        error_df = df[df.time.isin(error_times)]
        ax.plot(
            error_df.centroid_position.x,
            error_df.centroid_position.y,
            "x",
            color="red",
            markersize=8,
            alpha=0.8,
            zorder=10,
        )
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.4)


def plot_fov(
    ax,
    pos,
    head_deg,
    alpha_deg,
    max_dist=0.2,
    wedge_color="C0",
    wedge_alpha=0.21,
    arrow_kw=None,
):
    """ """

    x0, y0 = pos

    # Draw subject location
    ax.plot(x0, y0, "o", color=wedge_color, markersize=8, zorder=5)

    # Draw FOV wedge (matplotlib Wedge expects theta1, theta2 in degrees CCW from +x)
    half = alpha_deg / 2.0
    theta1 = head_deg - half
    theta2 = head_deg + half
    wedge = Wedge(
        (x0, y0),
        r=max_dist,
        theta1=theta1,
        theta2=theta2,
        facecolor=wedge_color,
        alpha=wedge_alpha,
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(wedge)

    # Draw head-direction arrow
    arrow_len = 0.5 * max_dist
    hd_rad = np.deg2rad(head_deg)
    dx = arrow_len * np.cos(hd_rad)
    dy = arrow_len * np.sin(hd_rad)
    if arrow_kw is None:
        arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, color=wedge_color, linewidth=1.5)
    arrow = FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy), **arrow_kw, zorder=6)
    ax.add_patch(arrow)


def get_first_goal_sight(trial_df, alpha_deg=120.0, smooth_SD=2, min_consecutive=1, max_distance=None):
    """
    come up with a way of defining when the goal is first seen on a given trial
    """
    a = trial_df.angle_to_goal.egocentric.values
    if smooth_SD:
        a = circular_gaussian(a, sigma=smooth_SD)
    inside = np.abs(a) <= (alpha_deg / 2.0)
    # optionally distance filter
    if max_distance is not None:
        d = trial_df.distance_to_goal.euclidean.values
        inside &= d <= max_distance

    min_frames = int(min_consecutive * FRMAE_RATE)
    # convolution trick to find runs of length >= min_consecutive
    conv = np.convolve(inside.astype(int), np.ones(min_frames, dtype=int), mode="valid")
    pos = np.flatnonzero(conv == min_frames)
    if pos.size == 0:
        return None, None
    first_idx = int(pos[0])  # start index of the run
    return first_idx, float(trial_df.iloc[first_idx][("time", "")])


def circular_gaussian(theta_deg, sigma=3):
    th = np.radians(theta_deg)
    x = np.cos(th)
    y = np.sin(th)

    xs = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    return np.degrees(np.arctan2(ys, xs))


# %% errors


def get_error_times(
    session,
    trial=1,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 120, "smooth_SD": 4, "min_consecutive": 0.2},
):
    """"""
    simple_maze = session.simple_maze()
    dec_df = _get_trial_df(
        session,
        trial,
        return_data_structure="decisions_df",
        first_goal_sight=first_goal_sight,
        goal_sight_kwargs=goal_sight_kwargs,
    )
    goal = dec_df.goal.unique()[0]

    # filter for decision points
    decision_points = get_decision_nodes(simple_maze)

    dec_df = dec_df[dec_df.maze_position.isin(decision_points)]
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    error_times = []
    for idx, row in dec_df.iterrows():
        pos = row.maze_position
        action = row.action
        optimal_actions = get_optimal_action(pos, goal, simple_maze, location_label2coord)
        error = action not in optimal_actions
        if error:
            error_times.append(row.time)

    return error_times


def get_decision_nodes(simple_maze):
    decision_coords = [node for node, deg in simple_maze.degree() if deg >= 3]
    node2label = nx.get_node_attributes(simple_maze, "label")
    return [node2label[coord] for coord in decision_coords]


def get_optimal_action(current_location, goal, simple_maze, location_label2coord=None):
    """
    Optimal choice defined as the option that decreases the shortest path distance to the goal the most.
    Note if multiple choice uptons have the same minimum shortest path distance to goal, all are considered optimal choices.
    """
    if location_label2coord is None:
        location_label2coord = {
            **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
            **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
        }
    current_coord = location_label2coord[current_location]
    goal_coord = location_label2coord[goal]
    neighbors = list(simple_maze.neighbors(current_coord))
    neighbor2geodesic_distance = {
        neighbor: nx.shortest_path_length(simple_maze, neighbor, goal_coord, weight="weight") for neighbor in neighbors
    }
    min_distance = min(neighbor2geodesic_distance.values())
    optimal_neighbors = [
        neighbor for neighbor, distance in neighbor2geodesic_distance.items() if distance == min_distance
    ]
    optimal_actions = [get_neighbor_cdir(current_coord, optimal_neighbor) for optimal_neighbor in optimal_neighbors]
    return optimal_actions


# %%


def get_excess_steps(
    session,
    trial,
    return_shortest_path=False,
    simple_maze=None,
    extended_maze=None,
    simple_label2coord=None,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.5},
):
    """ """
    # gen vars if not provided
    if simple_maze is None:
        simple_maze = session.simple_maze()
        if extended_maze is None:
            extended_maze = mr.get_extended_simple_maze(simple_maze)
    if simple_label2coord is None:
        simple_label2coord = mr.get_maze_label2coord(simple_maze)
        coord2simple_label = {v: k for k, v in simple_label2coord.items()}
    # filter for trial data
    trial_df = _get_trial_df(
        session,
        trial,
        return_data_structure="decisions_df",
        first_goal_sight=first_goal_sight,
        goal_sight_kwargs=goal_sight_kwargs,
    )
    traj = trial_df.maze_position.values
    start = traj[0]
    goal = trial_df.goal.unique()[0]
    shortest_path = nx.shortest_path(extended_maze, simple_label2coord[start], simple_label2coord[goal], weight=None)
    shortest_path_length = len(shortest_path)
    path_length = len(traj)
    n_excess_steps = path_length - shortest_path_length
    if return_shortest_path:
        shortest_path = [coord2simple_label[coord] for coord in shortest_path]
        return n_excess_steps, shortest_path
    else:
        return n_excess_steps


def _get_trial_df(
    session,
    trial,
    return_data_structure="decisions_df",
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.5},
    verbose=False,
):
    # load data
    decisions_df = session.trajectory_decisions_df
    navigation_df = session.navigation_df

    # filter for trial
    nav_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
    dec_df = decisions_df[(decisions_df.trial == trial) & (decisions_df.trial_phase == "navigation")]

    if first_goal_sight:
        first_idx, first_time = get_first_goal_sight(nav_df, **goal_sight_kwargs)
        if first_idx is not None:
            nav_df = nav_df[nav_df.time >= first_time]
            dec_df = dec_df[dec_df.time >= first_time]
    # check if data remaining (sometimes short trial)
    if len(nav_df) == 0 or len(dec_df) == 0:
        if verbose:
            print("No data remaining after applying first_goal_sight filter")
        return None
    # return requested data structure
    if return_data_structure == "decisions_df":
        return dec_df
    elif return_data_structure == "navigation_df":
        return nav_df
    else:
        raise ValueError("return_data_structure must be 'decisions_df' or 'navigation_df'")
