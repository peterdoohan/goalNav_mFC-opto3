""" """

# %% Imports
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import mixed_anova

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.strategies import models

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds


# %%%


def get_group_by_stim_strategy_weights(
    navigation_strategies_df,
    strategies=["vector", "structure", "habit", "backtracking_penalty", "forward_bias"],
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    max_trial_duration=None,
    stim_only=True,
):
    """ """
    # filter data
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if max_trial_duration is not None:
        keep_trials = df.groupby("trial_unique_ID").time_in_trial.max().le(max_trial_duration).index
        df = df[df.trial_unique_ID.isin(keep_trials)]
    if stim_only:
        # control trial times in non-stim times when filtering for stim_on
        # times only in stim trials
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]

    # fit nav strategy weights for stim_on and stim_off decisions per subject
    results = []
    for subject in SUBJECT_IDS:
        subj_df = df[df.subject_ID == subject]
        condition = subj_df.condition.unique()[0]
        for stim_trial in [True, False]:
            _df = subj_df[subj_df.stim_trial == stim_trial]
            if stim_trial and stim_only:
                _df = _df[_df.stim_on]
            # fit strategy weights on select data
            strategy_weights = models.get_navigation_strategy_weights(_df, strategies=strategies)
            results.append(
                {
                    "subject_ID": subject,
                    "condition": condition,
                    "stim_trial": stim_trial,
                    **strategy_weights,
                }
            )
    return pd.DataFrame(results)


# %% Old Functions


def plot_random_effects_summary(results_df, axes=None):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    for y, ax in zip(
        ["weight_vector", "weight_structure", "weight_penalty"],
        axes.flatten(),
    ):
        _plot_group_by_stim(results_df, y, ax=ax, print_stats=True)
    fig.tight_layout()


def _plot_group_by_stim(df, y, ax=None, print_stats=False):
    """ """
    # set up fig
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    conditions = ["control", "opto"]
    x_pos = {cond: i for i, cond in enumerate(conditions)}
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.4, len(conditions) - 0.6)
    ax.set_ylim(df[y].min() * 0.8, df[y].max() * 1.1)
    condition2color = {"control": "black", "opto": "#0077FF"}

    # plot subject-level paired points
    for cond in conditions:
        cond_df = df[df["condition"] == cond]
        cond_df = cond_df.set_index(["subject_ID", "stim_trial"])[y].unstack()
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
        order=conditions,
        y=y,
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
    ax.set_ylabel(y)
    if print_stats:
        stats_df = mixed_anova(
            dv=y,
            within="stim_trial",
            between="condition",
            subject="subject_ID",
            data=df,
        )
        print(stats_df)


def get_random_effects_summary_df(input_data, stim_day_range=(4, np.inf), stim_on_only=False, max_trial_duration=120):
    """ """
    # filter input data
    filtered_data = input_data.copy()
    if stim_day_range is not None:
        filtered_data = filtered_data[filtered_data.total_stim_days.between(*stim_day_range)]
    if max_trial_duration is not None:
        filtered_data = filtered_data[filtered_data.trial_duration.le(max_trial_duration)]
    # get strategy weights per subject
    results = []
    for subject in SUBJECT_IDS:
        df = filtered_data[filtered_data.subject_ID == subject]
        for stim_trial in [True, False]:
            _df = df[df.stim_trial == stim_trial]
            if stim_on_only:
                if stim_trial:
                    _df = _df[_df.stim_on]
                else:
                    # must match statistics between conditions when filtering for stim_on
                    _df = _df[_df.time_in_trial.le(MAX_STIM_DURATION)]
            # fit strategy weights on select data
            strategy_weights = models.get_navigation_strategy_weights(_df)
            strategy_weights["subject_ID"] = subject
            # strategy_weights["condition"] = SUBJECT_INFO.set_index("subject_ID").loc[subject].condition
            strategy_weights["stim_trial"] = stim_trial
            results.append(strategy_weights)

    return pd.DataFrame(results)


def get_input_data(
    subject_IDs="all",
    conditions="all",
    experiment_phases="expert",
    total_stim_days="all",
):
    """
    should add first goal sight filter to this...
    """
    sessions = gs.get_maze_sessions(
        subject_IDs=subject_IDs,
        conditions=conditions,
        experiment_phases=experiment_phases,
        total_stim_days=total_stim_days,
        with_data=["navigation_strategies_df", "trials_df"],
        must_have_data=True,
        verbose=True,
    )
    dfs = []
    for session in sessions:
        nav_strats_df = session.navigation_strategies_df
        # add trial durations for optional filtering later
        trials_df = session.trials_df
        trials_df.set_index("trial", inplace=True)
        trial_durations = (trials_df.time.trial_end - trials_df.time.cue).to_dict()
        nav_strats_df[("trial_duration", "")] = nav_strats_df.trial.map(trial_durations)
        # add total stim days
        nav_strats_df[("total_stim_days", "")] = session.total_stim_days
        dfs.append(nav_strats_df)
    input_df = pd.concat(dfs, axis=0, ignore_index=True)
    # filter out small amount of data where choice not defined (before cue)
    input_df = input_df[~input_df.choice_value.isna().all(axis=1)]
    return input_df
