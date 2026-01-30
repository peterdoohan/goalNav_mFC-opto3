"""
quantifty excess steps between stim conditions across groups
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from joblib import Parallel, delayed
from pingouin import mixed_anova
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import zscore, norm

from GridMaze.maze import representations as mr
from GridMaze.maze import metrics as mm
from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.behaviour import trajectory_plotting as tp
from scipy.spatial.distance import euclidean

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)


# %% Early stim effects


def plot_stim_effects_over_days(
    df,
    y="n_excess_steps",
    groups=["control", "opto"],
    stim_day_range=None,
    outlier_thres=500,
    ignore_sessions_with_issues_noted=False,
    rolling_avg=2,
    ax=None,
):
    """
    still need to find best way to plot
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)

    # filter data
    _df = df.copy()
    if outlier_thres is not None:
        _df = _df[_df[y] <= outlier_thres]
    if ignore_sessions_with_issues_noted:
        _df = _df[~_df.session_issue_noted]

    # average excess steps per subject per day
    df = _df.groupby(["condition", "subject_ID", "stim_trial", "total_stim_days"])[y].mean().reset_index()
    # pivot
    delta_steps = (
        df.pivot(index=["subject_ID", "condition", "total_stim_days"], columns="stim_trial", values=y)
        .diff(axis=1)[True]
        .reset_index()
    )
    delta_steps.rename(columns={True: f"delta_{y}"}, inplace=True)
    subject_grouped = delta_steps.groupby(["condition", "total_stim_days"])[f"delta_{y}"]
    mean_df = subject_grouped.mean().unstack(level=0)
    sem_df = subject_grouped.sem().unstack(level=0)
    if rolling_avg:
        mean_df = mean_df.rolling(window=rolling_avg, min_periods=1).mean()
        sem_df = sem_df.rolling(window=rolling_avg, min_periods=1).mean()
    days = mean_df.index.values
    # plot
    group2color = {"control": "grey", "opto": "#0077FF"}
    for cond in groups:
        color = group2color[cond]
        _means = mean_df[cond].values
        _sems = sem_df[cond].values
        ax.plot(days, _means, color=color, label=cond)
        ax.fill_between(
            days,
            (_means - _sems),
            (_means + _sems),
            color=color,
            alpha=0.2,
        )
    ax.legend(fontsize="x-small", frameon=False)
    ax.set_xlabel("stim day")
    ax.set_ylabel(f"Δ {y} \n (light on - light off)")
    if stim_day_range is not None:
        ax.set_xlim(stim_day_range)


# %% Group x stim summary and plotting functions


def plot_random_effects_summary(
    df,
    y="n_excess_steps",
    stim_day_range=(8, np.inf),
    outlier_thres=500,
    ignore_sessions_with_issues_noted=False,
    ignore_first_trial_after_stim=False,
    print_stats=True,
    stim_color="#0077FF",
    ax=None,
):
    """ """
    df = _get_group_by_stim_df(
        df,
        y=y,
        stim_day_range=stim_day_range,
        outlier_thres=outlier_thres,
        ignore_sessions_with_issues_noted=ignore_sessions_with_issues_noted,
        ignore_first_trial_after_stim=ignore_first_trial_after_stim,
    )

    # plot cross subject mean ± SEM
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    cp.plot_group_by_stim(df, y=y, ax=ax, stim_color=stim_color, print_stats=print_stats)


def _get_group_by_stim_df(
    df,
    y="n_excess_steps",
    stim_day_range=(6, np.inf),
    outlier_thres=500,
    ignore_sessions_with_issues_noted=False,
    ignore_first_trial_after_stim=False,
):
    # filter data
    _df = df.copy()
    if stim_day_range is not None:
        _df = _df[_df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        _df = _df[_df[y] <= outlier_thres]
    if ignore_first_trial_after_stim:
        _df = _df[_df.trials_since_stim != 1]
    if ignore_sessions_with_issues_noted:
        _df = _df[~_df.session_issue_noted]

    # average excess steps per subject over trials
    df = _df.groupby(["condition", "subject_ID", "stim_trial"])[y].mean().reset_index()
    return df


# %% excess steps functions


def get_performance_df(
    sessions=None,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.4},
    verbose=False,
    jobs=-1,
    save=False,
):
    """ """
    # load and return if already generated
    save_path = RESULTS_PATH / "behaviour" / "performance_df.parquet"
    if not save and save_path.exists():
        if verbose:
            print("Loading performance_df from results...")
        performance_df = pd.read_parquet(save_path)
        return performance_df

    # else generate
    if sessions is None:
        # load all stim sessions
        if verbose:
            print("Loading all stim sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            stim_only=True,
            with_data=["trials_df", "navigation_df", "trajectory_decisions_df"],
            must_have_data=True,
            verbose=True,
        )
    # calc excess steps & other metrics for each session
    if jobs:
        dfs = Parallel(n_jobs=jobs)(
            delayed(get_session_performance_df)(
                session,
                first_goal_sight=first_goal_sight,
                goal_sight_kwargs=goal_sight_kwargs,
            )
            for session in sessions
        )
    else:
        dfs = []
        for session in sessions:
            if verbose:
                print(session.name)
            _df = get_session_performance_df(
                session,
                first_goal_sight=first_goal_sight,
                goal_sight_kwargs=goal_sight_kwargs,
            )
            dfs.append(_df)
    performance_df = pd.concat(dfs, ignore_index=True)

    # save to disk
    if save:
        if verbose:
            print("Saving performance_df to results...")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_parquet(save_path)
    return performance_df


def get_session_performance_df(
    session,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.4},
    steps_as_nodes=True,
):
    """ """
    # load data
    simple_maze = session.simple_maze()
    skeleton_maze = session.skeleton_maze()
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    simple_label2coord = mr.get_maze_label2coord(simple_maze)
    simple_label2pos = mr.get_maze_label2position(simple_maze)
    skeleton_label2coord = mr.get_maze_label2coord(skeleton_maze)
    decisions_df = session.trajectory_decisions_df
    navigation_df = session.navigation_df
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)

    # calc excess steps for each trial
    trials = trials_df.index.unique()
    results = []
    for trial in trials:
        # filter for trial
        nav_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
        dec_df = decisions_df[(decisions_df.trial == trial) & (decisions_df.trial_phase == "navigation")]
        if len(nav_df) == 0 or len(dec_df) == 0:
            continue
        if first_goal_sight:
            # further filter out times before subject has seen goal
            first_idx, first_time = tp.get_first_goal_sight(nav_df, **goal_sight_kwargs)
            if first_idx is not None:
                nav_df = nav_df[nav_df.time >= first_time]
                dec_df = dec_df[dec_df.time >= first_time]
        if len(nav_df) == 0 or len(dec_df) == 0:
            continue
        # calculate excess steps
        traj = dec_df.maze_position.values
        node_list = [x for x in traj if not "-" in x]
        start = traj[0]
        first_node = node_list[0] if len(node_list) > 0 else np.nan
        goal = dec_df.goal.unique()[0]
        shortest_path = nx.shortest_path(
            extended_maze, simple_label2coord[start], simple_label2coord[goal], weight=None
        )
        shortest_path_length = len(shortest_path)
        path_length = len(traj)
        n_excess_steps = path_length - shortest_path_length
        # calculate useful variables for stratifying excess steps across trials
        start_pos = nav_df.iloc[0].centroid_position.values
        start_skel = nav_df.iloc[0].maze_position.skeleton
        start_euclidean_dist = euclidean(start_pos, simple_label2pos[goal])
        start_geodesic_dist = nx.shortest_path_length(
            skeleton_maze, skeleton_label2coord[start_skel], skeleton_label2coord[goal + "_C"], weight="weight"
        )
        # also calculate other performance metrics like trial_duration and n_errors (pokes into non-goal towers)
        n_errors = trials_df.loc[trial, ("errors", "")]
        trial_duration = trials_df.loc[trial, ("time", "reward")] - trials_df.loc[trial, ("time", "cue")]
        # store results
        results.append(
            {
                "subject_ID": session.subject_ID,
                "condition": session.condition,
                "maze_name": session.maze_name,
                "maze_order": session.maze_order,
                "day_on_maze": session.day_on_maze,
                "stim_day": session.stim_day,
                "total_stim_days": session.total_stim_days,
                "trial_unique_ID": nav_df.trial_unique_ID.unique()[0],
                "goal": goal,
                "stim_trial": trials_df.loc[trial, ("stim_trial", "")],
                "trials_since_stim": trials_df.loc[trial, ("trials_since_stim", "")],
                "consecutive_stim_trials": trials_df.loc[trial, ("consecutive_stim_trials", "")],
                "trial_duration": trial_duration,
                "n_errors": n_errors,
                "n_excess_steps": n_excess_steps,
                "shortest_path_length": shortest_path_length,
                "path_length": path_length,
                "start_location": start,
                "first_node": first_node,
                "start_euclidean_dist": start_euclidean_dist,
                "start_geodesic_dist": start_geodesic_dist,
            }
        )
    excess_steps_df = pd.DataFrame(results)

    # some sessions had low laser power where fiber was partially broken, keep note of this
    if session.session_notes is not None:
        issue_noted = True
    else:
        issue_noted = False
    excess_steps_df["session_issue_noted"] = issue_noted

    # optionally convert steps to nodes (instead of node+edge)
    if steps_as_nodes:
        excess_steps_df["n_excess_steps"] = excess_steps_df["n_excess_steps"] / 2

    return excess_steps_df
