""" """

# %% Imports
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import mixed_anova

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
    results_df = pd.DataFrame(results)
    return results_df


# %% Old Functions
