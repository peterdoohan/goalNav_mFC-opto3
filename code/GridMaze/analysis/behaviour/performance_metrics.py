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
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

# %% excess steps by goal


def test(
    excess_steps_df,
    stim_day_range=(4, np.inf),
    outlier_thres=500,
    var="betweenness_centrality",
    zscore_var=False,
):
    """ """
    # filter data
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]

    if var in ["start_euclidean_dist", "start_geodesic_dist", "start_dist_ratio"]:
        if var == "start_dist_ratio":
            v = df["start_geodesic_dist"] - df["start_euclidean_dist"]
        else:
            v = df[var]  # already computed

    else:
        # calculate var per trial
        maze_1 = mr.get_simple_maze("maze_1")
        maze_2 = mr.get_simple_maze("maze_2")
        if var == "betweenness_centrality":
            maze_1_dict = mm.get_betweeness_centrality(maze_1)
            maze_2_dict = mm.get_betweeness_centrality(maze_2)
        elif var == "mean_geodesic_distance":
            maze_1_dict = mm.get_mean_shortest_path_distance(maze_1)
            maze_2_dict = mm.get_mean_shortest_path_distance(maze_2)
        elif var == "node_degree":
            maze_1_dict = mm.get_node_degree(maze_1)
            maze_2_dict = mm.get_node_degree(maze_2)
        elif var == "mean_distance_decorrelation":
            maze_1_dict = mm.get_mean_distance_decorrelation(maze_1)
            maze_2_dict = mm.get_mean_distance_decorrelation(maze_2)
        else:
            raise NotImplementedError

        def _map_var(row):
            if row.maze_name == "maze_1":
                return maze_1_dict[row.goal]
            elif row.maze_name == "maze_2":
                return maze_2_dict[row.goal]

        v = df.apply(_map_var, axis=1)

    if zscore_var:
        v = zscore(v)
    df[var] = v

    # fit linear mixed effects model
    md = smf.mixedlm(
        f"n_excess_steps ~ condition * stim_trial * {var}", df, groups=df["subject_ID"], re_formula="~stim_trial"
    )
    res = md.fit(reml=False, method="lbfgs", maxiter=10_000)

    return res.summary()


def _get_mean_geodesic_distance(row):
    return


def plot_delta_delta_excess_steps_across_goals(
    excess_steps_df,
    maze_name="maze_2",
    stim_day_range=(4, np.inf),
    outlier_thres=500,
    highlight_significant=True,
    vmax=5,
    ax=None,
):
    """ """
    # filter data
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]
    df = df[df.maze_name == maze_name]
    # get delta (light on - light off) per goal per subject
    # then average this delta across subjects in each condition (opto/control)
    delta_df = (
        df.groupby(["subject_ID", "condition", "goal", "stim_trial"])
        .n_excess_steps.mean()
        .unstack(level=-1)
        .diff(axis=1)[True]
        .groupby(level=[1, 2])
        .mean()
        .unstack(level=0)
    )
    delta_delta_df = delta_df[("opto")] - delta_df[("control")]
    # stats (no exaclty correct, just placeholder approx.)
    if highlight_significant:
        # Fit mixed model (with subject random effects)
        md = smf.mixedlm(
            "n_excess_steps ~ condition * stim_trial * goal", df, groups=df["subject_ID"], re_formula="~stim_trial"
        )
        res = md.fit(reml=False, method="lbfgs", maxiter=10_000)
        goal2p_val = {}
        pvals = res.pvalues
        for goal in df.goal.unique():
            try:
                goal2p_val[goal] = pvals.loc[f"condition[T.opto]:stim_trial[T.True]:goal[T.{goal}]"]
            except KeyError:
                pass
        highlight_nodes = [goal for goal, p in goal2p_val.items() if p < 0.05]
    else:
        highlight_nodes = False

    # plotting
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
    simple_maze = mr.get_simple_maze(maze_name)
    edges = [l for l in mr.get_maze_locations(simple_maze) if "-" in l]  # add edge values as 0 for plotting
    mp.plot_simple_heatmap(
        simple_maze,
        pd.concat([delta_delta_df, pd.Series(0, index=edges)]),
        colormap="mako",
        highlight_nodes=highlight_nodes,
        value_label="ΔΔ excess steps",
        highlight_color="magenta",
        node_size=175,
        edge_size=6.5,
        vmin=0,
        vmax=vmax,
        ax=ax,
    )


# %% Early stim effects


def plot_stim_effects_over_days(
    excess_steps_df,
    groups=["control", "opto"],
    stim_day_range=None,
    outlier_thres=500,
    ignore_sessions_with_issues_noted=False,
    steps_as_nodes=True,
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
    _df = excess_steps_df.copy()
    if steps_as_nodes:
        _df["n_excess_steps"] = _df["n_excess_steps"] / 2  # convert steps to nodes
    if outlier_thres is not None:
        _df = _df[_df.n_excess_steps <= outlier_thres]
    if ignore_sessions_with_issues_noted:
        _df = _df[~_df.session_issue_noted]

    # average excess steps per subject per day
    df = _df.groupby(["condition", "subject_ID", "stim_trial", "total_stim_days"]).n_excess_steps.mean().reset_index()
    # pivot
    delta_steps = (
        df.pivot(index=["subject_ID", "condition", "total_stim_days"], columns="stim_trial", values="n_excess_steps")
        .diff(axis=1)[True]
        .reset_index()
    )
    delta_steps.rename(columns={True: "delta_excess_steps"}, inplace=True)
    subject_grouped = delta_steps.groupby(["condition", "total_stim_days"]).delta_excess_steps
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
    ax.set_ylabel("Δ excess steps \n (light on - light off)")
    if stim_day_range is not None:
        ax.set_xlim(stim_day_range)


# %% Group x stim summary and plotting functions


def plot_random_effects_summary(
    df,
    y="n_excess_steps",
    stim_day_range=(8, np.inf),
    outlier_thres=500,
    starting_dist_range=None,
    ignore_sessions_with_issues_noted=False,
    ignore_first_trial_after_stim=False,
    steps_as_nodes=True,
    print_stats=True,
    ax=None,
):
    """ """
    # filter data
    _df = df.copy()
    if stim_day_range is not None:
        _df = _df[_df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        _df = _df[_df[y] <= outlier_thres]
    if starting_dist_range is not None:
        _df = _df[_df.start_geodesic_dist.between(*starting_dist_range)]
    if ignore_first_trial_after_stim:
        _df = _df[_df.trials_since_stim != 1]
    if ignore_sessions_with_issues_noted:
        _df = _df[~_df.session_issue_noted]

    # average excess steps per subject over trials
    df = _df.groupby(["condition", "subject_ID", "stim_trial"])[y].mean().reset_index()
    if y == "n_excess_steps" and steps_as_nodes:
        df["n_excess_steps"] = df["n_excess_steps"] / 2  # convert steps to nodes

    # plot cross subject mean ± SEM
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    cp.plot_group_by_stim(df, y=y, ax=ax, print_stats=print_stats)


# %% excess steps functions


def get_performance_df(
    sessions=None,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.4},
    verbose=True,
    jobs=-1,
):
    """ """
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
    # calc excess steps for each session
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
    excess_steps_df = pd.concat(dfs, ignore_index=True)
    return excess_steps_df


def get_session_performance_df(
    session,
    first_goal_sight=True,
    goal_sight_kwargs={"alpha_deg": 160, "smooth_SD": 4, "min_consecutive": 0.4},
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
        start = traj[0]
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

    return excess_steps_df
