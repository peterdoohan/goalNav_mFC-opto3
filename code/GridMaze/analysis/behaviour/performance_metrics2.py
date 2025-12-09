""" """

# %% Imports
import json
from tkinter import font
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import representations as mr

# %% Global Variables

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% testing


def test(df, y="duration", xrange=(0, 50)):
    f, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for cond, ax in zip(["control", "opto"], axes):
        for stim in [True, False]:
            subset = df[(df.condition == cond) & (df.stim_trial == stim)]
            sns.histplot(
                x=subset[y].values,
                fill=True,
                alpha=0.5,
                stat="density",
                element="step",
                label=f"stim: {stim}",
                ax=ax,
                binrange=xrange,
            )
        ax.set_xlabel(cond)
    ax.legend(fontsize=8)
    return


# %% plotting


def _plot_median_excess_steps(df, ax=None, print_stats=True):
    xs_long = df.groupby(["condition", "subject_ID", "stim_trial"]).n_excess_steps.mean().reset_index()
    _plot_x_across_conditions(xs_long, "n_excess_steps", ax=ax)
    # stats
    if print_stats:
        pass
    return


def _plot_median_trial_duration(df, ax=None, print_stats=True):
    """ """
    dur_long = df.groupby(["condition", "subject_ID", "stim_trial"]).duration.mean().reset_index()
    _plot_x_across_conditions(dur_long, "duration", ax=ax)
    # stats
    if print_stats:
        pass
    return


def _plot_x_across_conditions(df_long, x, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 4))
    ax.spines[["top", "right"]].set_visible(False)
    # single subjects
    sns.stripplot(
        data=df_long,
        x="condition",
        y=x,
        hue="stim_trial",
        dodge=True,
        jitter=True,
        alpha=0.75,
        legend=False,
        ax=ax,
    )
    # mean ± SEM
    sns.pointplot(
        data=df_long,
        x="condition",
        y=x,
        hue="stim_trial",
        dodge=0.2,
        linestyle="none",
        errorbar="se",
        ax=ax,
    )


# %%
def test(df):
    xs_long = df.groupby(["condition", "subject_ID", "stim_trial"]).n_excess_steps.mean().reset_index()
    plot_x_across_groups(xs_long, "n_excess_steps", print_stats=True)
    return


def plot_x_across_groups(df, x, print_stats=True):
    conditions = ["opto", "control"]
    x_pos = {c: i for i, c in enumerate(conditions)}

    # fixed style settings
    color_off = "black"
    color_on = "#0077FF"  # electric blue
    grey = "lightgrey"
    mean_pt_size = 8

    fig, ax = plt.subplots(1, 1, figsize=(2, 3))

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.4, len(conditions) - 0.6)

    ymax = df[x].max() * 1.2
    ax.set_ylim(0, ymax)

    # plot subject-level paired points
    for cond in conditions:
        subdf = df[df["condition"] == cond]
        for subj, sdf in subdf.groupby("subject_ID"):
            sdf = sdf.sort_values("stim_trial")
            x_off = x_pos[cond] - 0.20
            x_on = x_pos[cond] + 0.20
            y_off = sdf[sdf["stim_trial"] == False][x].iloc[0]
            y_on = sdf[sdf["stim_trial"] == True][x].iloc[0]
            ax.plot([x_off, x_on], [y_off, y_on], "-", color=grey, lw=1.5, alpha=0.8)

        # plot mean ± SEM
        for stim_val, col in [(False, color_off), (True, color_on)]:
            for cond in conditions:
                vals = (
                    df[(df["condition"] == cond) & (df["stim_trial"] == stim_val)]
                    .groupby("subject_ID")[x]
                    .first()
                    .dropna()
                    .values
                )
                mean = vals.mean()
                sem = stats.sem(vals)
                x = x_pos[cond] - 0.20 if not stim_val else x_pos[cond] + 0.20
                ax.errorbar(x, mean, yerr=sem, fmt="o", color=col, ms=mean_pt_size, capsize=0, elinewidth=3)

    # simple legend (means only)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_off, label="stim_off", markersize=9),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_on, label="stim_trial", markersize=9),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize="small", loc="lower left")

    # shared labels
    fig.text(0.5, 0.02, "condition", ha="center", fontsize=13)
    fig.text(0.03, 0.5, "weight", va="center", rotation="vertical", fontsize=13)

    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    if print_stats:
        model = smf.mixedlm("n_excess_steps ~ condition * stim_trial", data=df, groups="subject_ID").fit()
        print(model.summary())


# %%


def filter_basic_behaviour_df(
    basic_behaviour_df,
    stim_sessions=True,
    ignore_first_n_stim_sessions=False,
    max_trial_duration=None,  # seconds (ignore trials longer than this -- off task)
):
    # filter data based on input
    df = basic_behaviour_df.copy()
    if stim_sessions:
        df = df[df.stim_session == True]
    if ignore_first_n_stim_sessions:
        assert isinstance(ignore_first_n_stim_sessions, int)
        df = df[df.stim_day > ignore_first_n_stim_sessions]
    if max_trial_duration is not None:
        df = df[df.duration <= max_trial_duration]

    return df


# %% get basic behaviour


def get_analysis_sessions():
    sessions = gs.get_maze_sessions(
        subject_IDs="all",
        conditions="all",
        with_data=["trials_df", "navigation_df", "trajectory_decisions_df"],
        must_have_data=True,
        verbose=True,
    )
    return sessions


def get_basic_behaviour_df(sessions=None, save=False):
    save_path = RESULTS_PATH / "behaviour" / "performance_metrics" / "basic_behaviour_df.htsv"
    if save_path.exists():
        return pd.read_csv(save_path, sep="\t")
    else:
        if sessions is None:
            sessions = get_analysis_sessions()
        session_results = []
        for session in sessions:
            print(session.name)
            trials_df = session.trials_df
            trial_durations = (trials_df.time.reward - trials_df.time.cue).to_numpy()
            n_excess_steps = get_n_excess_steps(session)
            path_distance_at_cue = get_path_distance_to_goal_at_cue(session)
            stim_trial = [True if x != False else False for x in trials_df.stim_trial]
            results_df = pd.DataFrame(
                {
                    "subject_ID": session.subject_ID,
                    "maze_name": session.maze_name,
                    "experiment_day": session.experimental_day,
                    "condition": session.condition,
                    "tethered": session.tethered,
                    "stim_session": session.stim,
                    "stim_day": session.stim_day,
                    "stim_trial": stim_trial,
                    "trial": trials_df.trial,
                    "goal": trials_df.goal,
                    "errors": trials_df.errors,
                    "duration": trial_durations,
                    "n_excess_steps": n_excess_steps,
                    "path_distance_at_cue": path_distance_at_cue,
                }
            )
            session_results.append(results_df)
        combined_results = pd.concat(session_results, axis=0)
        # save
        if save:
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            combined_results.to_csv(save_path, sep="\t", index=False)
        return combined_results


def get_n_excess_steps(session):
    """ """
    decisions_df = session.trajectory_decisions_df  # cleaned up node-edges transitions (backtracking corrections)
    #
    simple_maze = session.simple_maze()
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    simple_label2coord = mr.get_maze_label2coord(simple_maze)
    #
    trials = session.trials_df.trial.to_numpy()
    excess_steps = []
    for trial in trials:
        trial_df = decisions_df[decisions_df.trial == trial]
        if len(trial_df) <= 1:
            excess_steps.append(np.nan)
            continue  # NaN value if naviagtion contains no transitions between nodes
        path = trial_df.maze_position
        start = path.iloc[0]
        goal = trial_df.goal.unique()[0]
        shortest_path = (
            nx.shortest_path_length(extended_maze, simple_label2coord[start], simple_label2coord[goal], weight=None) - 1
        )
        path_length = len(path) - 1
        n_excess_steps = path_length - shortest_path
        excess_steps.append(n_excess_steps)

    return excess_steps


def get_path_distance_to_goal_at_cue(session):
    """ """
    simple_maze = session.simple_maze()
    extended_maze = mr.get_extended_simple_maze(simple_maze)
    navigation_df = session.navigation_df
    trials = session.trials_df.trial.to_numpy()
    simple_label2coord = mr.get_maze_label2coord(simple_maze)
    distances = []
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        start = trial_df.maze_position.simple.iloc[0]
        goal = trial_df.goal.unique()[0]
        dist = nx.shortest_path_length(extended_maze, simple_label2coord[start], simple_label2coord[goal], weight=None)
        distances.append(dist)
    return distances


# %% Functions
