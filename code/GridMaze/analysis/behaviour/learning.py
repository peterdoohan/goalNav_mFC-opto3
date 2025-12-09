"""
Compare learning curves across opto and control groups (no stim)
@peterdoohan
"""

# %% Import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from scipy.stats import ttest_ind


from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
SESSION_DURATION = 40

# %% Functions


def plot_performance_summary(learning_df, expert_df, maze_order=1, print_stats=True, axes=None):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(3, 3), width_ratios=[1, 0.3], sharey=True)

    plot_learning_summary(learning_df, maze_order=maze_order, print_stats=print_stats, ax=axes[0])
    plot_expert_performance_summary(
        expert_df, maze_order=maze_order, print_stats=print_stats, y_axis_off=True, ax=axes[1]
    )


def plot_learning_summary(learning_curve_df, maze_order=1, print_stats=True, ax=None):
    """ """
    # set up figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)

    # filter data
    df = learning_curve_df.copy()
    df = df[df.maze_order == maze_order]

    # plot
    sns.lineplot(
        data=df,
        x="day_on_maze",
        y="total_trials",
        hue="condition",
        palette=["#0077FF", "black"],
        errorbar="se",
        ax=ax,
    )
    ax.set_xlabel("day on maze")
    ax.set_ylabel("total trials")
    sns.move_legend(
        ax,
        title=None,
        loc="upper left",
        fontsize="small",
    )
    ymax = df.total_trials.max() * 1.1
    ax.set_ylim(0, ymax)
    ax.set_xlabel("learning days")
    ax.set_ylabel("trials")

    # stats
    if print_stats:
        print("Learning stats:")
        model = mixedlm("total_trials ~ condition * day_on_maze", df, groups=df["subject_ID"])
        result = model.fit(reml=False)
        print(result.summary())


def get_learning_curve_df(sessions=None, verbose=False):
    """ """
    # load data
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            experiment_phases=["learning"],
            with_data=["trials_df", "events_df"],
            must_have_data=True,
        )
    # get n. trials per session
    results = []
    for session in sessions:
        if verbose:
            print(session.name)
        # check session did not finish early (pycontrol error)
        if not _check_session_duration(session):
            continue
        total_trials = session.trials_df.trial.max()
        results.append(
            {
                "subject_ID": session.subject_ID,
                "condition": session.condition,
                "maze_order": session.maze_order,
                "day_on_maze": session.day_on_maze,
                "total_trials": total_trials,
            }
        )
    learning_curve_df = pd.DataFrame(results)
    return learning_curve_df


def _check_session_duration(session, duration_buffer=5):
    # check session did not finish early (pycontrol error)
    end_time = session.events_df.time.max() / 60  # mins
    if not (SESSION_DURATION - duration_buffer) <= end_time <= (SESSION_DURATION + duration_buffer):
        return False
    else:
        return True


# %% summaries expert data


def plot_expert_performance_summary(
    expert_performance_df, maze_order=1, stim_day_range=None, print_stats=True, y_axis_off=False, ax=None
):
    """ """
    # set up fig
    if ax is None:
        f, ax = plt.subplots(figsize=(1, 3))
    ax.spines[["top", "right"]].set_visible(False)

    # filter data
    df = expert_performance_df.copy()
    df = df[df.maze_order == maze_order]
    if stim_day_range is not None:
        df = df[df.stim_day.between(*stim_day_range)]

    # avg trials per subject over expert behaviour
    subj_av = df.groupby(["subject_ID", "condition"]).total_trials.mean().reset_index()

    # plot individual sessions per condition
    sns.stripplot(
        data=df,
        y="total_trials",
        hue="condition",
        palette=["#0077FF", "black"],
        alpha=0.05,
        dodge=0.01,
        legend=False,
    )
    # plot subject mean & sem
    sns.pointplot(
        data=subj_av,
        y="total_trials",
        hue="condition",
        palette=["#0077FF", "black"],
        errorbar="se",
        dodge=0.4,
        ax=ax,
        legend=False,
    )
    ax.set_ylabel("trials")
    ax.set_xlabel("expert \n sessions")

    if y_axis_off:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)

    # stats
    if print_stats:
        # non-paired t-test
        print("Expert performance stats:")
        control_trials = subj_av[subj_av.condition == "control"].total_trials
        opto_trials = subj_av[subj_av.condition == "opto"].total_trials
        t_stat, p_val = ttest_ind(control_trials, opto_trials)
        print(f"random effects t-test: t={t_stat:.2f}, p={p_val:.3f}")


def get_expert_performance_df(sessions=None, verbose=False, ignore_low_laser_power_sessions=False):
    """ """
    # load data
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            experiment_phases=["expert"],
            with_data=["trials_df", "events_df"],
            must_have_data=True,
        )
    # get n. trials per session
    results = []
    for session in sessions:
        if verbose:
            print(session.name)
        # check session duration
        if not _check_session_duration(session):
            continue
        # filter out sessions where fiber was damaged / laser power low
        if ignore_low_laser_power_sessions:
            if session.session_notes is not None and "laser power low" in session.session_notes:
                continue
        total_trials = session.trials_df.trial.max()
        results.append(
            {
                "subject_ID": session.subject_ID,
                "condition": session.condition,
                "maze_order": session.maze_order,
                "stim_day": session.stim_day,
                "total_trials": total_trials,
            }
        )
    expert_performance_df = pd.DataFrame(results)
    return expert_performance_df
