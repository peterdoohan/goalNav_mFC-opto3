"""
This module generates a navigation stratagies df that can be used to model maze navigation behaviour as a function of vector navigation and
structure navigation (model-based) stragegies. These dataframes are to be populated in the analysis data folder and loaded as session object attributes
"""

# %% imports
import numpy as np
import pandas as pd
import networkx as nx
from GridMaze.maze import representations as mr
from GridMaze.analysis.core import get_sessions as gs

# %% Global variables

# %% session level navigation strategies df


def get_session_navigation_strategies_df(
    session,
    strategies=["vector", "structure", "backtracking_penalty", "forward_bias", "habits"],
    habits_n_back=3,
    remove_edge_backtracks=True,
    ignore_final_step=True,
):
    # get initalised sd
    init_df = get_init_df(
        session,
        remove_edge_backtracks=remove_edge_backtracks,
        ignore_final_step=ignore_final_step,
    )
    # get further variables
    simple_maze = session.simple_maze()
    node2action_available = get_node2action_available(simple_maze)
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    coord2pos = nx.get_node_attributes(simple_maze, "position")
    all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze, weight="weight"))
    opp_actions = {"N": "S", "S": "N", "E": "W", "W": "E"}

    # define general mapping function
    def _get_values(row, strategy):
        if strategy == "subject_choices":
            return get_subject_choices(row[("action", "")])
        elif strategy == "optimal_actions":
            return get_optimal_actions(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                simple_maze=simple_maze,
                all_shortest_path_lengths=all_shortest_path_lengths,
            )
        elif strategy == "available":
            return get_available(
                row[("location", "")],
                node2action_available=node2action_available,
            )
        elif strategy == "vector":
            return get_vector_values(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                coord2pos=coord2pos,
            )
        elif strategy == "structure":
            return get_structure_values(
                row[("location", "")],
                row[("goal", "")],
                label2coord=label2coord,
                simple_maze=simple_maze,
                all_shortest_path_lengths=all_shortest_path_lengths,
            )
        elif strategy == "backtracking_penalty":
            return get_backtracking_penalty_values(
                row[("previous_action", "")],
                opp_actions=opp_actions,
            )
        elif strategy == "forward_bias":
            return get_forward_bias_values(row[("previous_action", "")])
        elif strategy == "habits":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # add subject_choice, optimal_choice and availability info to df

    # add further strategies requested from input


def get_init_df(
    session,
    remove_edge_backtracks=True,
    ignore_final_step=True,
):
    """
    initalise navigation_strategies_df with node transitions defined trial by trial
    with goal and other info specified too, that can use used to define strategies
    for modelling subject's choices
    """
    # load data
    navigation_df = session.navigation_df
    session_info = session.session_info
    simple_maze = session.simple_maze()
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    # further process navigation_df
    label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    navigation_df[("maze_position", "simple_shifted")] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    trials = navigation_df.trial.unique()
    # loop over trials
    dfs = []
    for t in trials:
        trial_df = navigation_df[navigation_df.trial == t]
        if trial_df.empty:
            continue
        start_time = trial_df.iloc[0].time.values[0]
        # filter transitions between nodes
        transitions_df = trial_df[trial_df.maze_position.simple_change]
        transitions_df = transitions_df[transitions_df.maze_position.simple.apply(lambda x: len(x.split("-")) == 1)]
        transitions_df = transitions_df[transitions_df.cardinal_movement_direction.notnull()].reset_index(drop=True)
        locs = transitions_df.maze_position.simple
        actions = transitions_df.cardinal_movement_direction
        prev_actions = actions.shift(1)
        times = transitions_df.time
        if remove_edge_backtracks:
            # node transitions can be double counted if a mouse backtacks on an edge or if mouse position oscillates between
            # an adjacent edge and node, if desired, remove these backtrack to give very clean trajectory transitions and choices
            backtrack_mask = locs == locs.shift(1)
            locs = locs[~backtrack_mask]
            actions = pd.Series(get_trajectory_actions(locs, label2coord=label2coord))
            prev_actions = actions.shift(1)
            times = times[~backtrack_mask]
        if ignore_final_step:
            locs, actions, prev_actions, times = locs[:-1], actions[:-1], prev_actions[:-1], times[:-1]
            if locs.empty:
                continue
        # build df
        _df = pd.DataFrame(index=locs.index)
        _df[("trial", "")] = t
        _df[("trial_unique_ID", "")] = gs.get_session_name(session_info) + f"_trial{t}"
        _df[("time_in_trial", "")] = times.sub(start_time)
        _df[("goal", "")] = trials_df.loc[t].goal
        _df[("stim_trial", "")] = trials_df.loc[t].stim_trial
        _df[("stim_on", "")] = choice_time2stim_on(trials_df.reset_index(), times)
        _df[("location", "")] = locs.values
        _df[("action", "")] = actions.values
        _df[("previous_action", "")] = prev_actions.values
        _df[("nth_visit", "")] = locs.to_frame().groupby("simple").cumcount()
        dfs.append(_df)
    # combine with session level info
    init_df = pd.concat(dfs, ignore_index=True)
    info_df = pd.DataFrame(
        {
            ("subject_ID", ""): session.subject_ID,
            ("condition", ""): session.condition,
            ("maze_name", ""): session.maze_name,
            ("day_on_maze", ""): session.day_on_maze,
            ("stim_day", ""): session.stim_day,
            ("total_stim_days", ""): session.total_stim_days,
        },
        index=init_df.index,
    )
    return pd.concat([info_df, init_df], axis=1)


# %% strategy functions (all return dict of option values for N/S/E/W)


def get_available(loc, node2action_available=None, simple_maze=None):
    if node2action_available is None:
        assert simple_maze is not None
        node2action_available = get_node2action_available(simple_maze)
    return node2action_available[loc]


def get_subject_choices(action):
    return {cdir: 0 if cdir != action else 1 for cdir in ["N", "S", "E", "W"]}


def get_optimal_actions(loc, goal, label2coord=None, simple_maze=None, all_shortest_path_lengths=None):
    """
    Optimal choice defined as the option that decreases the shortest path distance to the goal the most.
    Note if multiple choice uptons have the same minimum shortest path distance to goal, all are considered optimal choices.
    """
    # check inputs
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if all_shortest_path_lengths is None:
        assert simple_maze is not None, "Either simple_maze or all_shortest_path_lengths must be provided"
        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze, weight="weight"))

    loc_coord = label2coord[loc]
    goal_coord = label2coord[goal]
    neighbors = list(simple_maze.neighbors(loc_coord))
    neighbor2geodesic_distance = {neighbor: all_shortest_path_lengths[neighbor][goal_coord] for neighbor in neighbors}
    min_distance = min(neighbor2geodesic_distance.values())
    optimal_neighbors = [
        neighbor for neighbor, distance in neighbor2geodesic_distance.items() if distance == min_distance
    ]
    optimal_choices = [get_neighbor_cdir(loc_coord, optimal_neighbor) for optimal_neighbor in optimal_neighbors]
    return {cdir: 1 if cdir in optimal_choices else 0 for cdir in ["N", "S", "E", "W"]}


def get_vector_values(loc, goal, label2coord=None, coord2pos=None, simple_maze=None):
    """
    Returns a dictionary of vector navigation values for each available direction from the current location to the goal.
    Navigation values are calcuated as the cosine of the allocentric angle to goal (close to 1 if travleing in direction of goal,
    close to -1 if traveling away from goal, 0 if traveling perpendicular to goal).

    Note: this function ignores whether a direction is available or not, it just returns the vector navigation value for each direction.
    This is accounted for later by specifiying which options are available in the navigation strategies dataframe.
    """
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if coord2pos is None:
        assert simple_maze is not None, "Either simple_maze or coord2pos must be provided"
        coord2pos = nx.get_node_attributes(simple_maze, "position")

    coord = label2coord[loc]
    pos = coord2pos[coord]
    goal_pos = coord2pos[label2coord[goal]]
    goal_angle = np.arctan2(goal_pos[0] - pos[0], goal_pos[1] - pos[1])  # in radians from pos y = true north
    if goal_angle < 0:
        goal_angle = 2 * np.pi + goal_angle
    return {
        "N": np.cos(goal_angle),
        "S": np.cos(goal_angle + np.pi),
        "E": np.cos(goal_angle - np.pi / 2),
        "W": np.cos(goal_angle + np.pi / 2),
    }


def get_structure_values(loc, goal, label2coord=None, simple_maze=None, all_shortest_path_lengths=None):
    """
    Returns a dictionary of structure navigation values for each available direction from the current location to the goal.
    Options are assigned avalue of 1 if they decrease the shortest path distance to the goal, -1 if they increase the shortest
    path distance to the goal, and 0 if shortest-path distances remains unchanged.

    Note this function isgnores whether options/directions are available at each location, however shortest-path distance calucaltions
    do repsect maze structure. This is accounted for later by specifiying which options are available in the navigation strategies dataframe.
    """
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
        label2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    if all_shortest_path_lengths is None:
        assert simple_maze is not None, "Either simple_maze or all_shortest_path_lengths must be provided"
        all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(simple_maze, weight="weight"))

    coord = label2coord[loc]
    x, y = coord
    goal_coord = label2coord[goal]
    current_distance_to_goal = all_shortest_path_lengths[coord][goal_coord]
    neighbours = [  # need to be irrespective of maze structure
        (x, y + 1),
        (x, y - 1),
        (x + 1, y),
        (x - 1, y),
    ]
    neighbours = [node for node in neighbours if node in simple_maze.nodes()]  # remove nodes that off the maze
    values = {}
    for neigbour in neighbours:
        neigbour_cdir = get_neighbor_cdir(coord, neigbour)
        neigbour_geodesic_distance_to_goal = all_shortest_path_lengths[neigbour][goal_coord]
        if current_distance_to_goal > neigbour_geodesic_distance_to_goal:
            values[neigbour_cdir] = 1
        elif current_distance_to_goal < neigbour_geodesic_distance_to_goal:
            values[neigbour_cdir] = -1
        else:
            values[neigbour_cdir] = 0
    invalid_actions = list(set(["N", "S", "E", "W"]) - set(values.keys()))
    for invalid_cdir in invalid_actions:  # going off the maze is always long shortest path
        values[invalid_cdir] = -1
    return values


def get_backtracking_penalty_values(prev_action, opp_actions=None):
    if opp_actions is None:
        opp_actions = {"N": "S", "S": "N", "E": "W", "W": "E"}
    return {cdir: -1 if cdir == opp_actions.get(prev_action) else 0 for cdir in ["N", "S", "E", "W"]}


def get_forward_bias_values(prev_action):
    return {cdir: 1 if cdir == prev_action else 0 for cdir in ["N", "S", "E", "W"]}


# %% strategy utility functions
def get_node2action_available(simple_maze):
    """
    Returns a dict of the available directions at each node in the maze.
    The keys are the available directions ('N', 'S', 'E', 'W') and the
    values are True if the direction is available and False otherwise.
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node2NSEW_available = {}
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        node_NSEW2available = {"N": False, "S": False, "E": False, "W": False}
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                node_NSEW2available["E"] = True
            if neighbor[0] == node[0] - 1:
                node_NSEW2available["W"] = True
            if neighbor[1] == node[1] + 1:
                node_NSEW2available["N"] = True
            if neighbor[1] == node[1] - 1:
                node_NSEW2available["S"] = True
        node2NSEW_available[node_coord2label[node]] = node_NSEW2available
    return node2NSEW_available


def get_neighbor_cdir(location_coord, neigbour_coord):
    """Returns the cardinal direction of the neighbour relative to the current location."""
    if location_coord[0] == neigbour_coord[0] + 1:
        return "W"
    elif location_coord[0] == neigbour_coord[0] - 1:
        return "E"
    elif location_coord[1] == neigbour_coord[1] + 1:
        return "S"
    elif location_coord[1] == neigbour_coord[1] - 1:
        return "N"


# %% Main function


def get_navigation_strategies_df(session, remove_edge_backtracks=True):
    """
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing decision values for each navigation choice made by the mouse in the session. The DataFrame has the following columns:
        - subject_ID: the ID of the mouse
        - maze_name: the name of the maze
        - day_on_maze: the day on which the maze was run
        - trial: the trial number
        - goal: the goal location
        - current_location: the current location of the mouse
        - nth_visit: the nth visit to the current location
        - vector_navigation_value_N/S/E/W: the vector navigation value for each next choice (direction)
        - structure_navigation_value_N/S/E/W: the structure navigation value for each next choice (direction)
        - penalty_value_N/S/E/W: the penalty value for each next choice, penalties are (currently) only applied to the last visited location/direction
        - choice_value_N/S/E/W: the actual choice value for each direction (1 for choice value, others = 0)
        - available_N/S/E/W: whether each direction is available from the current location (defined by maze structure)

    """
    # load maze and navigation df
    trials_df = session.trials_df
    navigation_df = session.navigation_df
    session_info = session.session_info
    trials = [t for t in navigation_df.trial.unique() if not np.isnan(t)]
    trial2goal = navigation_df.set_index("trial").goal.dropna().to_dict()
    simple_maze = mr.simple_maze(session_info["maze_structure"])
    # process data
    node2NSEW_available = get_node2NSEW_available(simple_maze)
    navigation_df[("maze_position", "simple_shifted")] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    trial_decision_values_dfs = []
    # set up decision df columns
    columns = pd.MultiIndex.from_tuples(
        [
            ("subject_ID", ""),
            ("condition", ""),
            ("maze_name", ""),
            ("day_on_maze", ""),
            ("stim", ""),
            ("stim_day", ""),
            ("trial", ""),
            ("trial_unique_ID", ""),
            ("stim_trial", ""),
            ("time_in_trial", ""),
            ("goal", ""),
            ("stim_on", ""),
            ("current_location", ""),
            ("nth_visit", ""),
            *pd.MultiIndex.from_product(
                [
                    [
                        "vector_navigation_value",
                        "structure_navigation_value",
                        "penalty_value",
                        "choice_value",
                        "optimal_choice_value",
                        "available",
                    ],
                    ["N", "S", "E", "W"],
                ]
            ).tolist(),
        ]
    )
    for trial in trials:
        navigation_trial_df = navigation_df[navigation_df.trial == trial]
        goal = trial2goal[trial]
        trial_stim_type = trials_df[trials_df.trial == trial].stim_trial.values[0]
        if (
            len(navigation_trial_df) == 0
        ):  # if no navigation data for trial (same reward location activated during rewared consumption)
            trial_decision_values_df = _get_empty_decision_values_df(
                trial, goal, session_info, trial_stim_type, columns
            )
            trial_decision_values_dfs.append(trial_decision_values_df)
            continue
        trial_start_time = navigation_trial_df.iloc[0].time.values[0]
        trial_transitions_df = navigation_trial_df[navigation_trial_df.maze_position.simple_change]
        # some transitions oscilated between adjacent locations over sequential frames (cardinal direction = None/NaN), exclude these:
        trial_transitions_df = trial_transitions_df[trial_transitions_df.cardinal_movement_direction.notnull()]
        trial_trajectory = trial_transitions_df.maze_position.simple
        trial_choices = trial_transitions_df.cardinal_movement_direction
        choice_times = trial_transitions_df.time
        nodes_mask = trial_trajectory.apply(lambda x: len(x.split("-")) == 1)
        trial_node_trajectory = trial_trajectory[nodes_mask]
        trial_node_choices = trial_choices[nodes_mask]
        node_choice_times = choice_times[nodes_mask]
        if remove_edge_backtracks:
            # node transitions can be double counted if a mouse backtacks on an edge or if mouse position oscillates between
            # an adjacent edge and node, if desired, remove these backtrack to give very clean trajectory transitions and choices
            node_duplication_mask = trial_node_trajectory == trial_node_trajectory.shift(1)
            trial_node_trajectory = trial_node_trajectory[~node_duplication_mask]
            trial_node_choices = pd.Series(get_trajectory_actions(trial_node_trajectory, simple_maze))
            node_choice_times = node_choice_times[~node_duplication_mask]
        trial_node_trajectory = trial_node_trajectory[:-1].reset_index(drop=True)  # last node is at reward
        trial_node_choices = trial_node_choices[:-1].reset_index(drop=True)  # no choice to be made at reward
        node_choice_times = node_choice_times[:-1].reset_index(drop=True)
        choice_stim_on = choice_time2stim_on(trials_df, node_choice_times)  # if stim was on at time of choice
        nth_visit = trial_node_trajectory.to_frame().groupby("simple").cumcount()
        if len(trial_node_trajectory) == 0:  # trial started close to reward (no navigation)
            trial_decision_values_df = _get_empty_decision_values_df(
                trial, goal, session_info, trial_stim_type, columns
            )
            trial_decision_values_dfs.append(trial_decision_values_df)
            continue
        trial_decision_values_df = pd.DataFrame(
            columns=columns,
            data=np.full((len(trial_node_trajectory), len(columns)), np.nan),
        )
        trial_decision_values_df[("subject_ID", "")] = session_info["subject_ID"]
        trial_decision_values_df[("condition", "")] = session_info["condition"]
        trial_decision_values_df[("maze_name", "")] = session_info["maze_name"]
        trial_decision_values_df[("day_on_maze", "")] = session_info["day_on_maze"]
        trial_decision_values_df[("stim_day", "")] = session_info["stim_day"]
        trial_decision_values_df[("stim", "")] = session_info["stim"]
        trial_decision_values_df[("trial", "")] = trial
        trial_decision_values_df[("trial_unique_ID", "")] = gs.get_session_name(session_info) + f"_trial{trial}"
        trial_decision_values_df[("stim_trial", "")] = bool(trial_stim_type)
        trial_decision_values_df[("time_in_trial", "")] = node_choice_times.sub(trial_start_time)
        trial_decision_values_df[("goal", "")] = goal
        trial_decision_values_df[("stim_on", "")] = choice_stim_on
        trial_decision_values_df[("current_location", "")] = trial_node_trajectory
        trial_decision_values_df[("nth_visit", "")] = nth_visit
        for i, current_location in enumerate(trial_node_trajectory):
            choice = {cdir: 0 if cdir != trial_node_choices[i] else 1 for cdir in ["N", "S", "E", "W"]}
            optimal_choice = get_location2optimal_choice_value(current_location, goal, simple_maze)
            vector_nav_option_values = get_location2option_vector_navigation_values(current_location, goal, simple_maze)
            structure_nav_option_values = get_location2option_structure_navigation_values(
                current_location, goal, simple_maze
            )
            if i == 0:
                penalty_values = {
                    "N": 0,
                    "S": 0,
                    "E": 0,
                    "W": 0,
                }  # avoid penalising first move, in future could keep track of choice before navigation to avoid this
            else:
                opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
                previous_choice = trial_node_choices[i - 1]
                penalty_values = {cdir: -1 if cdir == opp.get(previous_choice) else 0 for cdir in ["N", "S", "E", "W"]}
            available_values = node2NSEW_available[current_location]
            for option in ["N", "S", "E", "W"]:
                trial_decision_values_df.loc[i, ("vector_navigation_value", option)] = vector_nav_option_values[option]
                trial_decision_values_df.loc[i, ("structure_navigation_value", option)] = structure_nav_option_values[
                    option
                ]
                trial_decision_values_df.loc[i, ("penalty_value", option)] = penalty_values[option]
                trial_decision_values_df.loc[i, ("choice_value", option)] = choice[option]
                trial_decision_values_df.loc[i, ("optimal_choice_value", option)] = optimal_choice[option]
                trial_decision_values_df.loc[i, ("available", option)] = int(available_values[option])
        trial_decision_values_dfs.append(trial_decision_values_df)
        navigation_strategies_df = pd.concat(trial_decision_values_dfs, ignore_index=True)
        # convert binary valeus in stim_on and available to bool (conserving nans)
        navigation_strategies_df[("stim_on", "")] = navigation_strategies_df[("stim_on", "")].map(_to_bool)
        for d in ["N", "S", "E", "W"]:
            navigation_strategies_df[("available", d)] = navigation_strategies_df[("available", d)].map(_to_bool)
    return navigation_strategies_df


# %% utility functions


def _to_bool(x):
    """Convers binary to bool while retainings nans"""
    if pd.isna(x):
        return x
    return bool(x)


def choice_time2stim_on(trials_df, node_choice_times):
    """"""
    for _, t in trials_df.iterrows():
        stim_on = node_choice_times.gt(t.time.stim_start) & node_choice_times.lt(t.time.stim_end)
        if stim_on.any():
            return stim_on
    # if no stim on times found, return all False
    stim_on = np.full(len(node_choice_times), False)
    return stim_on


def get_trajectory_actions(node_traj, simple_maze=None, label2coord=None):
    if label2coord is None:
        assert simple_maze is not None, "Either simple_maze or label2coord must be provided"
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


def _get_empty_decision_values_df(trial, goal, session_info, stim_type, columns):
    """
    Returns an empty trial_decision_values_df for trials where mouse started close to reward (no navigation).
    Still contains session and trial info for consistency
    """
    trial_decision_values_df = pd.DataFrame(columns=columns, data=np.full((1, len(columns)), np.nan))
    trial_decision_values_df[("subject_ID", "")] = session_info["subject_ID"]
    trial_decision_values_df[("condition", "")] = session_info["condition"]
    trial_decision_values_df[("maze_name", "")] = session_info["maze_name"]
    trial_decision_values_df[("stim", "")] = session_info["stim"]
    trial_decision_values_df[("day_on_maze", "")] = session_info["day_on_maze"]
    trial_decision_values_df[("stim_day", "")] = session_info["stim_day"]
    trial_decision_values_df[("trial", "")] = trial
    trial_decision_values_df[("trial_unique_ID", "")] = gs.get_session_name(session_info) + f"_trial{trial}"
    trial_decision_values_df[("stim_trial", "")] = bool(stim_type)
    trial_decision_values_df[("time_in_trial", "")] = 0
    trial_decision_values_df[("goal", "")] = goal
    return trial_decision_values_df


# %% Supporting functions


def get_location2option_vector_navigation_values(current_location, goal, simple_maze):
    """
    Returns a dictionary of vector navigation values for each available direction from the current location to the goal.
    Navigation values are calcuated as the cosine of the allocentric angle to goal (close to 1 if travleing in direction of goal,
    close to -1 if traveling away from goal, 0 if traveling perpendicular to goal).

    Note: this function ignores whether a direction is available or not, it just returns the vector navigation value for each direction.
    This is accounted for later by specifiying which options are available in the navigation strategies dataframe.

    Parameters
    ----------
    current_location : str
        The label of the current location in the maze.
    goal : str
        The label of the goal location in the maze.
    simple_maze : networkx.Graph
        The maze as a networkx graph.

    Returns
    -------
    dict
        A dictionary containing the vector navigation value for each available direction from the current location to the goal.
        The keys are the available directions ('N', 'S', 'E', 'W') and the values are the corresponding vector navigation values.

    """
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }

    location_coord2position = {
        **nx.get_node_attributes(simple_maze, "position"),
        **nx.get_edge_attributes(simple_maze, "position"),
    }

    current_coord = location_label2coord[current_location]
    current_position = location_coord2position[current_coord]
    goal_position = location_coord2position[location_label2coord[goal]]
    goal_angle = np.arctan2(
        goal_position[0] - current_position[0], goal_position[1] - current_position[1]
    )  # in radians from pos y = true north
    if goal_angle < 0:
        goal_angle = 2 * np.pi + goal_angle
    NSEW2navigation_values = {}
    for neighbour_cdir in ["N", "S", "E", "W"]:
        if neighbour_cdir == "N":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle)
        elif neighbour_cdir == "S":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle + np.pi)
        elif neighbour_cdir == "E":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle - np.pi / 2)
        elif neighbour_cdir == "W":
            NSEW2navigation_values[neighbour_cdir] = np.cos(goal_angle + np.pi / 2)
    return NSEW2navigation_values


def get_location2option_structure_navigation_values(current_location, goal, simple_maze):
    """
    Returns a dictionary of structure navigation values for each available direction from the current location to the goal.
    Options are assigned avalue of 1 if they decrease the shortest path distance to the goal, -1 if they increase the shortest
    path distance to the goal, and 0 if shortest-path distances remains unchanged.

    Note this function isgnores whether options/directions are available at each location, however shortest-path distance calucaltions
    do repsect maze structure. This is accounted for later by specifiying which options are available in the navigation strategies dataframe.

    Parameters
    ----------
    current_location : str
        The label of the current location in the maze.
    goal : str
        The label of the goal location in the maze.
    simple_maze : networkx.Graph
        The maze as a networkx graph.

    Returns
    -------
    dict
        A dictionary containing the structure navigation value for each available direction from the current location to the goal.
        The keys are the available directions ('N', 'S', 'E', 'W') and the values are the corresponding structure navigation values.
    """
    location_label2coord = {
        **{v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()},
        **{v: k for k, v in nx.get_edge_attributes(simple_maze, "label").items()},
    }
    current_coord = location_label2coord[current_location]
    current_x, current_y = current_coord
    goal_coord = location_label2coord[goal]
    current_geodesic_distance_to_goal = nx.shortest_path_length(simple_maze, current_coord, goal_coord, weight="weight")
    # neighbors = list(simple_maze.neighbors(current_coord))
    neighbours = [
        (current_x, current_y + 1),
        (current_x, current_y - 1),
        (current_x + 1, current_y),
        (current_x - 1, current_y),
    ]
    neighbours = [node for node in neighbours if node in simple_maze.nodes()]  # remove nodes that are not in the maze
    NSEW2navigation_values = {}
    for neigbour in neighbours:
        neigbour_cdir = get_neighbor_cdir(current_coord, neigbour)
        neigbour_geodesic_distance_to_goal = nx.shortest_path_length(simple_maze, neigbour, goal_coord, weight="weight")
        if current_geodesic_distance_to_goal > neigbour_geodesic_distance_to_goal:
            NSEW2navigation_values[neigbour_cdir] = 1
        elif current_geodesic_distance_to_goal < neigbour_geodesic_distance_to_goal:
            NSEW2navigation_values[neigbour_cdir] = -1
        else:
            NSEW2navigation_values[neigbour_cdir] = 0
    invalid_cdirs = list(set(["N", "S", "E", "W"]) - set(NSEW2navigation_values.keys()))
    for invalid_cdir in invalid_cdirs:  # going off the maze is always long shortest path
        NSEW2navigation_values[invalid_cdir] = -1
    return NSEW2navigation_values


def get_location2optimal_choice_value(current_location, goal, simple_maze):
    """
    Optimal choice defined as the option that decreases the shortest path distance to the goal the most.
    Note if multiple choice uptons have the same minimum shortest path distance to goal, all are considered optimal choices.
    """
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
    optimal_choices = [get_neighbor_cdir(current_coord, optimal_neighbor) for optimal_neighbor in optimal_neighbors]
    return {cdir: 1 if cdir in optimal_choices else 0 for cdir in ["N", "S", "E", "W"]}


def get_location2positions(simple_maze):
    """Returns a dict of the positions (x,y in meters) of each location in the maze."""
    node_label2coord = nx.get_node_attributes(simple_maze, "label")
    node_label2position = nx.get_node_attributes(simple_maze, "position")
    node_coord2position = {node_label2coord[node]: node_label2position[node] for node in node_label2coord}
    edge_label2coord = nx.get_edge_attributes(simple_maze, "label")
    edge_label2position = nx.get_edge_attributes(simple_maze, "position")
    edge_coord2position = {edge_label2coord[edge]: edge_label2position[edge] for edge in edge_label2coord}
    return {**node_coord2position, **edge_coord2position}


def get_node2NSEW_available(simple_maze):
    """
    Returns a dict of the available directions at each node in the maze.
    The keys are the available directions ('N', 'S', 'E', 'W') and the
    values are True if the direction is available and False otherwise.
    """
    node_coord2label = nx.get_node_attributes(simple_maze, "label")
    node2NSEW_available = {}
    for node in simple_maze.nodes:
        neighbors = list(simple_maze.neighbors(node))
        node_NSEW2available = {"N": False, "S": False, "E": False, "W": False}
        for neighbor in neighbors:
            if neighbor[0] == node[0] + 1:
                node_NSEW2available["E"] = True
            if neighbor[0] == node[0] - 1:
                node_NSEW2available["W"] = True
            if neighbor[1] == node[1] + 1:
                node_NSEW2available["N"] = True
            if neighbor[1] == node[1] - 1:
                node_NSEW2available["S"] = True
        node2NSEW_available[node_coord2label[node]] = node_NSEW2available
    return node2NSEW_available
