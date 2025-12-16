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
from patsy import build_design_matrices

from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.behaviour import trajectory_plotting as tp
from scipy.spatial.distance import euclidean

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

# %% excess steps by goal


def test(
    excess_steps_df,
    maze_name="maze_2",
    stim_day_range=(4, np.inf),
    outlier_thres=600,
    highlight_significant=True,
):
    """ """
    # filter data
    df = excess_steps_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        df = df[df.n_excess_steps <= outlier_thres]
    df = df[df.maze_name == maze_name]
    # fixed effects for plotting (get av n_excess_steps per condition/goal/stim_trial)
    goal_xs_steps = df.groupby(["condition", "stim_trial", "goal"]).n_excess_steps.mean().unstack(level=[0, 1])
    goal_xs_steps = (
        df.groupby(["subject_ID", "condition", "stim_trial", "goal"])
        .n_excess_steps.mean()
        .groupby(["condition", "stim_trial", "goal"])
        .mean()
        .unstack([0, 1])
    )
    delta_df = pd.DataFrame(index=goal_xs_steps.index, columns=["control", "opto"])
    delta_df["control"] = goal_xs_steps[("control", True)] - goal_xs_steps[("control", False)]
    delta_df["opto"] = goal_xs_steps[("opto", True)] - goal_xs_steps[("opto", False)]
    delta_delta = delta_df.opto - delta_df.control

    # random effects for stats
    df["zscore_excess_steps"] = zscore(df["n_excess_steps"])

    # Fit mixed model
    md = smf.mixedlm(
        "zscore_excess_steps ~ condition * stim_trial * goal", df, groups=df["subject_ID"], re_formula="~stim_trial"
    )
    # md = smf.mixedlm(  # need to check model, is using some werid contrast thing...
    #     "zscore_excess_steps ~ condition * stim_trial * C(goal, Sum())",
    #     df,
    #     groups=df["subject_ID"],
    #     re_formula="~stim_trial",
    # )
    res = md.fit(reml=False, method="lbfgs", maxiter=10_000)
    goals = goal_xs_steps.index.values
    goal2p_val = {}
    pvals = res.pvalues
    for goal in goals:
        try:
            # goal2p_val[goal] = pvals.loc[f"condition[T.opto]:stim_trial[T.True]:C(goal, Sum())[S.{goal}]"]
            goal2p_val[goal] = pvals.loc[f"condition[T.opto]:stim_trial[T.True]:goal[T.{goal}]"]

        except KeyError:
            pass
    if highlight_significant:
        highlight_nodes = [goal for goal, p in goal2p_val.items() if p < 0.05]
    else:
        highlight_nodes = False
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    simple_maze = mr.get_simple_maze(maze_name)
    mp.plot_simple_heatmap(
        simple_maze,
        delta_delta,
        colormap="viridis",
        highlight_nodes=highlight_nodes,
        highlight_color="red",
        ax=ax,
    )

    return res


# %% Early stim effects


def plot_stim_effects_over_days(
    excess_steps_df,
    groups=["control", "opto"],
    stim_day_range=None,
    outlier_thres=500,
    ignore_low_laser_power_sessions=False,
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
    if ignore_low_laser_power_sessions:
        _df = _df[~_df.low_laser_power]

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
    excess_steps_df,
    stim_day_range=(8, np.inf),
    outlier_thres=500,
    starting_dist_range=None,
    ignore_low_laser_power_sessions=False,
    ignore_first_trial_after_stim=False,
    steps_as_nodes=True,
    print_stats=True,
    ax=None,
):
    """ """
    # filter data
    _df = excess_steps_df.copy()
    if stim_day_range is not None:
        _df = _df[_df.total_stim_days.between(*stim_day_range)]
    if outlier_thres is not None:
        _df = _df[_df.n_excess_steps <= outlier_thres]
    if starting_dist_range is not None:
        _df = _df[_df.start_geodesic_dist.between(*starting_dist_range)]
    if ignore_low_laser_power_sessions:
        _df = _df[~_df.low_laser_power]
    if ignore_first_trial_after_stim:
        _df = _df[_df.trials_since_stim != 1]

    # average excess steps per subject over trials
    df = _df.groupby(["condition", "subject_ID", "stim_trial"]).n_excess_steps.mean().reset_index()
    if steps_as_nodes:
        df["n_excess_steps"] = df["n_excess_steps"] / 2  # convert steps to nodes

    # set up fig
    conditions = ["control", "opto"]
    x_pos = {cond: i for i, cond in enumerate(conditions)}
    y_max = df.n_excess_steps.max() * 1.1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.4, len(conditions) - 0.6)
    ax.set_ylim(0, y_max)
    condition2color = {"control": "black", "opto": "#0077FF"}

    # plot subject-level paired points
    for cond in conditions:
        cond_df = df[df["condition"] == cond]
        cond_df = cond_df.set_index(["subject_ID", "stim_trial"]).n_excess_steps.unstack()
        x_off = x_pos[cond] - 0.30
        x_on = x_pos[cond] + 0.30
        for subj, row in cond_df.iterrows():
            y_off = row[False]
            y_on = row[True]
            ax.plot([x_off, x_on], [y_off, y_on], "-", color="lightgrey", lw=1.5, alpha=0.8)

    # plot cross subject mean ± SEM
    sns.pointplot(
        data=df,
        x="condition",
        y="n_excess_steps",
        hue="stim_trial",
        dodge=0.3,
        linestyle="none",
        errorbar="se",
        palette=[condition2color[c] for c in conditions],
        ax=ax,
    )
    sns.move_legend(
        ax,
        "lower center",
        ncol=2,
        title="light on",
        frameon=True,
        fontsize="x-small",
    )
    ax.set_xlabel("group")
    ax.set_ylabel("excess steps")

    if print_stats:
        stats_df = mixed_anova(
            dv="n_excess_steps",
            within="stim_trial",
            between="condition",
            subject="subject_ID",
            data=df,
        )
        # extract and display relevant stats
        cond = stats_df.loc[stats_df["Source"] == "condition"].iloc[0]
        stim = stats_df.loc[stats_df["Source"] == "stim_trial"].iloc[0]
        inter = stats_df.loc[stats_df["Source"] == "Interaction"].iloc[0]
        textstr = (
            f"Group: p={cond['p-unc']:.3f}\n" f"Stim  : p={stim['p-unc']:.3f}\n" f"Int     : p={inter['p-unc']:.3f}\n"
        )
        ax.text(
            0.05,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
        )
        print(stats_df)


# %% excess steps functions


def get_excess_steps_df(
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
            delayed(get_session_excess_steps_df)(
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
            _df = get_session_excess_steps_df(
                session,
                first_goal_sight=first_goal_sight,
                goal_sight_kwargs=goal_sight_kwargs,
            )
            dfs.append(_df)
    excess_steps_df = pd.concat(dfs, ignore_index=True)
    return excess_steps_df


def get_session_excess_steps_df(
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
                "n_excess_steps": n_excess_steps,
                "shortest_path_length": shortest_path_length,
                "path_length": path_length,
                "start_euclidean_dist": start_euclidean_dist,
                "start_geodesic_dist": start_geodesic_dist,
            }
        )
    excess_steps_df = pd.DataFrame(results)

    # some sessions had low laser power where fiber was partially broken, keep note of this
    if session.session_notes is not None and "laser power low" in session.session_notes:
        low_laser_power = True
    else:
        low_laser_power = False
    excess_steps_df["low_laser_power"] = low_laser_power

    return excess_steps_df
