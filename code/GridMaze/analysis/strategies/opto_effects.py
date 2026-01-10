""" """

# %% Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.strategies import models

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds


# %%%


def plot_mixture_of_strategy_weights(
    results_df,
    cmap="tab10",
    print_stats=True,
    axes=None,
):
    """ """
    strategies = [c for c in results_df.columns if c not in ["subject_ID", "condition", "stim_trial"]]
    if axes is None:
        fig, axes = plt.subplots(1, len(strategies), figsize=(2 * len(strategies), 3.5))
    for ax in axes:
        ax.axhline(0, color="k", ls="--", alpha=0.5)

    colors = sns.color_palette(cmap, len(strategies))
    for strategy, color, ax in zip(strategies, colors, axes):
        cp.plot_group_by_stim(
            results_df,
            y=strategy,
            ax=ax,
            stim_color=color,
            print_stats=print_stats,
        )


def get_group_by_stim_strategy_weights(
    navigation_strategies_df,
    strategies=["vector", "structure", "habit", "backtracking_penalty"],
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    max_trial_duration=None,
    stim_only=True,
    subsample_non_stim_trials=False,
):
    """ """
    # filter data
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if max_trial_duration is not None:
        keep_trials = df.groupby("trial_unique_ID").time_in_trial.max().le(max_trial_duration).index
        df = df[df.trial_unique_ID.isin(keep_trials)]
    if subsample_non_stim_trials:
        # more data in non-stim condition, optionally balance data in conditions
        # to check for sampling bias effects
        stim_trials = df[df.stim_trial].trial_unique_ID.to_list()
        keep_non_stim_trials = _subsample_non_stim_trials(df)
        df = df[df.trial_unique_ID.isin(stim_trials + keep_non_stim_trials)]
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
    results_df = pd.DataFrame(results)
    return results_df


# %% data subsampling functions


def _subsample_non_stim_trials(navigation_strategies_df, seed=0):
    """
    subsamples non-stim trials to match the number of stim trials
    stratified by subject and goal to ensure this is balanced across conditions
    """
    df = navigation_strategies_df.copy()
    unique_trials_df = df[["subject_ID", "goal", "stim_trial", "trial_unique_ID"]].drop_duplicates()
    stim_df = unique_trials_df[unique_trials_df.stim_trial].droplevel(1, axis=1)
    norm_df = unique_trials_df[~unique_trials_df.stim_trial].droplevel(1, axis=1)
    stim_trial_counts = stim_df.groupby(["subject_ID", "goal"]).size().reset_index(name="trial_counts")

    trials_with_counts = norm_df.merge(stim_trial_counts, on=["subject_ID", "goal"], how="inner")

    sampled_df = (
        trials_with_counts.groupby(["subject_ID", "goal"], group_keys=False)
        .apply(lambda g: g.sample(n=int(g["trial_counts"].iloc[0]), random_state=seed), include_groups=False)
        .reset_index(drop=True)
    )
    return sampled_df.trial_unique_ID.tolist()
