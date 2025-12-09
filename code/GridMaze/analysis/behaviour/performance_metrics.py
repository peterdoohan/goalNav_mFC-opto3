"""
Library for the analysis of basic performance metrics during the goalNav Big Maze task.
@ peterdoohan
"""

# %% Imports
import json
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import ttest_rel

from ..core import get_sessions as gs

# %% Global Variables
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")
# %% Functions


def get_sessions_for_analysis(experiment_phase="learning"):
    """ """
    sessions = gs.get_sessions(
        experiment_phases=experiment_phase,
        with_data=["trials_df", "navigation_df"],
    )
    return sessions


def get_basic_behaviour_summary_df(sessions):
    dfs = []
    for session in sessions:
        trials_df = session.trials_df
        navigation_duration = trials_df.time.reward - trials_df.time.cue
        navigation_efficencies = get_navigation_efficencies(session, metric="excess_steps")
        basic_behaviour_df = pd.DataFrame(
            {
                "subject_ID": session.subject_ID,
                "condition": session.condition,
                "sex": session.sex,
                "maze_name": session.maze_name,
                "experiment_phase": session.experiment_phase,
                "learning_day": session.learning_day,
                "stim_day": session.stim_day,
                "stim_type": session.stim_type,
                "trial": trials_df.trial,
                "stim_trial": trials_df.stim_trial,
                "goal": trials_df.goal,
                "errors": trials_df.errors,
                "navigation_duration": navigation_duration,
                "n_excess_steps": navigation_efficencies,
            }
        )
        dfs.append(basic_behaviour_df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def get_navigation_efficencies(session, metric="excess_steps"):
    if metric not in ["excess_steps", "fraction_of_optimal"]:
        raise ValueError("metric must be 'excess_path_length' or 'fraction_of_optimal'")
    simple_maze = session.simple_maze()
    simple_node2coord = {v: k for k, v in nx.get_node_attributes(simple_maze, "label").items()}
    navigation_df = session.navigation_df
    trials = [t for t in navigation_df.trial.unique() if not np.isnan(t)]
    navigation_df[("maze_position", "simple_shifted")] = navigation_df.maze_position.simple.shift(1)
    navigation_df[("maze_position", "simple_change")] = (
        navigation_df.maze_position.simple != navigation_df.maze_position.simple_shifted
    )
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    navigation_df = navigation_df[navigation_df.maze_position.simple_change == True]
    trial_efficencies = []
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        if len(trial_df) == 0:
            trial_efficencies.append(np.nan)
            continue  # NaN value if naviagtion contains no transitions between nodes
        goal = trial_df.goal.unique()[0]
        goal_coord = simple_node2coord[goal]
        full_path = trial_df.maze_position.simple
        node_path = full_path[full_path.apply(lambda x: len(x.split("-")) == 1)].to_numpy()  # remove edges
        node_path = np.array([key for key, _ in itertools.groupby(node_path)])  # remove duplicates
        if not len(node_path) < 2:
            path = [simple_node2coord[node] for node in node_path]  # convert to coordinates
            if metric == "fraction_of_optimal":
                shortest_path = nx.shortest_path_length(simple_maze, path[0], goal_coord, weight="weight")
                path_length = sum(simple_maze[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))
                trial_efficency = shortest_path / path_length
            if metric == "excess_steps":
                shortest_path = nx.shortest_path_length(simple_maze, path[0], goal_coord, weight=None)
                path_length = len(path) - 1
                trial_efficency = path_length - shortest_path
            trial_efficencies.append(trial_efficency)
        else:  # NaN value is trials start right next to goal
            trial_efficencies.append(np.nan)
    return trial_efficencies


# %% plotting
def plot_basic_behaviour_metrics(experiment_phase="learning", ax=None):
    """ """
    sessions = gs.get_sessions(
        experiment_phases=experiment_phase,
        with_data=["trials_df", "navigation_df"],
    )
    basic_behaviour_df = get_basic_behaviour_summary_df(sessions)
    if experiment_phase == "learning":
        day = "learning_day"
    elif experiment_phase == "full_trial_stim":
        day = "stim_day"
    subject_day_grouped_df = basic_behaviour_df.groupby(["subject_ID", "condition", "sex", day])
    # set up figure
    if ax is None:
        fig = plt.figure(figsize=(4, 4), clear=True)
        gspec = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gspec[0, 0])  # total trials
        ax2 = fig.add_subplot(gspec[0, 1])  # trial durations
        ax3 = fig.add_subplot(gspec[1, 0])  # trial errors
        ax4 = fig.add_subplot(gspec[1, 1])  # n excess steps
        axes = [ax1, ax2, ax3, ax4]
        fig.tight_layout()
    # plot metrics across days
    max_trial_grouped_df = subject_day_grouped_df["trial"].max().reset_index()
    median_duration_grouped_df = subject_day_grouped_df["navigation_duration"].median().reset_index()
    av_errors_grouped_df = subject_day_grouped_df["errors"].mean().reset_index()
    n_excess_steps_grouped_df = subject_day_grouped_df["n_excess_steps"].median().reset_index()
    grouped_dfs = [
        max_trial_grouped_df,
        median_duration_grouped_df,
        av_errors_grouped_df,
        n_excess_steps_grouped_df,
    ]
    ylabels = [
        "Trials",
        "Trial Duration (s)",
        "n Errors",
        "n Excess Steps",
    ]
    y_var = [
        "trial",
        "navigation_duration",
        "errors",
        "n_excess_steps",
    ]
    for grouped_df, ylabel, y, ax in zip(grouped_dfs, ylabels, y_var, axes):
        for cond, color in zip(["opto", "control"], ["deepskyblue", "grey"]):
            legend = cond if ax == ax4 else None
            condition_df = grouped_df[grouped_df.condition == cond]
            sns.lineplot(
                x=day,
                y=y,
                color=color,
                data=condition_df,
                errorbar="sd",
                err_style="band",
                ax=ax,
                label=legend,
            )
            for sex, linestyle in zip(["male", "female"], ["-", "--"]):
                condition_sex_df = condition_df[condition_df.sex == sex]
                sns.lineplot(
                    x=day,
                    y=y,
                    color=color,
                    data=condition_sex_df,
                    units="subject_ID",
                    linewidth=0.5,
                    linestyle=linestyle,
                    alpha=0.5,
                    estimator=None,
                    legend=False,
                    ax=ax,
                )
        ax.set_xlim(1, basic_behaviour_df[day].max())
        ax.set_xlabel("Days")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# %% stim plotting functions


def test(stim_sessions):
    stim_behaviour_df = get_basic_behaviour_summary_df(stim_sessions)
    subjects = stim_behaviour_df.subject_ID.unique()
    stim_trials_df = stim_behaviour_df[stim_behaviour_df.stim_trial == "full_trial"]
    no_stim_trials_df = stim_behaviour_df[stim_behaviour_df.stim_trial == False]
    results_df = []
    for subject in subjects:
        for df, bool in zip([stim_trials_df, no_stim_trials_df], [True, False]):
            subject_df = df[df.subject_ID == subject]
            results_df.append(
                {
                    "subject_ID": subject,
                    "stim": bool,
                    "mean_error_pokes": subject_df.errors.mean(),
                    "mean_trial_duration": subject_df.navigation_duration.mean(),
                    "mean_n_excess_steps": subject_df.n_excess_steps.mean(),
                }
            )
    return pd.DataFrame(results_df)


def plot_locomotion_metrics():
    """ """
    for subject in SUBJECT_IDS:
        sessions = gs.get_sessions(subject_IDs=[subject], experiment_phases="full_trial_stim")
        if len(sessions) == 0:
            continue
        navigation_df = pd.concat([session.navigation_df for session in sessions], axis=0)
        navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
        stim_on_df = navigation_df[navigation_df.stim_on == True]
        stim_off_df = navigation_df[navigation_df.stim_on == False]
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.histplot(
            stim_on_df.speed,
            color="deepskyblue",
            # kde=True,
            ax=ax,
            label="stim on",
            log_scale=True,
            fill=True,
            stat="density",
            element="step",
        )
        sns.histplot(
            stim_off_df.speed,
            color="grey",
            # kde=True,
            ax=ax,
            label="stim off",
            log_scale=True,
            fill=True,
            stat="density",
            element="step",
        )
    return


def get_average_speed(sessions):
    navigation_df = pd.concat([session.navigation_df for session in sessions], axis=0)
    navigation_df = navigation_df[navigation_df.trial_phase == "navigation"]
    stim_on_av_speed = navigation_df[navigation_df.stim_on == True].speed.mean()
    stim_off_av_speed = navigation_df[navigation_df.stim_on == False].speed.mean()
    return stim_on_av_speed, stim_off_av_speed


def plot_within_subject_stim_metrics_comparisons(group="opto", f=None, axes=None):
    """ """
    results_df = []
    valid_subjects = SUBJECT_INFO_DF[
        (SUBJECT_INFO_DF.included_in_full_trial_stim) & (SUBJECT_INFO_DF.condition == group)
    ].subject_ID.to_list()
    for subject in valid_subjects:
        sessions = gs.get_sessions(subject_IDs=[subject], experiment_phases="full_trial_stim")
        stim_on_av_speed, stim_off_av_speed = get_average_speed(sessions)
        stim_behaviour_df = get_basic_behaviour_summary_df(sessions)
        stim_trials_df = stim_behaviour_df[stim_behaviour_df.stim_trial == "full_trial"]
        no_stim_trials_df = stim_behaviour_df[stim_behaviour_df.stim_trial == False]
        for df, av_speed, bool in zip(
            [stim_trials_df, no_stim_trials_df], [stim_on_av_speed, stim_off_av_speed], [True, False]
        ):
            results_df.append(
                {
                    "subject_ID": subject,
                    "stim_on": bool,
                    "median_error_pokes": df.errors.median(),
                    "median_trial_duration": df.navigation_duration.median(),
                    "median_n_excess_steps": df.n_excess_steps.median(),
                    "av_speed": av_speed,
                }
            )
    results_df = pd.DataFrame(results_df)
    # plotting
    if axes is None or f is None:
        f, axes = plt.subplots(2, 2, figsize=(4, 4), sharex=True, clear=True)
    dodge = 0.3 if group == "opto" else False
    for var, ax, label in zip(
        results_df.columns[2:].to_numpy(), axes.flatten(), ["n Errors", "Trial Duration", "n Excess Steps", "Av Speed"]
    ):
        # stats
        if not group == "control":  # one animal in control group
            t, p = ttest_rel(results_df[results_df.stim_on][var], results_df[~results_df.stim_on][var])
            ax.set_title(f"T={t:.3f} p={p:.3f}", size=10)
        # plotting
        sns.pointplot(
            data=results_df,
            x="stim_on",
            y=var,
            ax=ax,
            hue="subject_ID",
            palette="tab10",
            alpha=1,
            legend=False,
            dodge=dodge,
        )
        if var == "median_error_pokes":
            ax.set_ylim(-1, 5)
        elif var == "median_trial_duration":
            ax.set_ylim(0, 50)
        elif var == "median_n_excess_steps":
            ax.set_ylim(-2, 20)
        elif var == "av_speed":
            ax.set_ylim(0, 0.3)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    f.tight_layout()
