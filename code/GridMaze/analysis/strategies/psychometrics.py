"""
Make some psycometric curves for P(correct) vs Habit Values
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.strategies import get_input_data as gid


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %% Functions


def get_psychometrics_df(
    navigation_strategies_df,
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    stim_only=True,
):
    """ """
    # filter data
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]

    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            subj_df = df[(df.subject_ID == subject) & (df.stim_trial == stim_trial)]
            return subj_df
            condition = subj_df.condition.unique()[0]

    return


def get_psychometric_curve(df, x="habit", bins=5):
    """ """
    # add column for if subject choice == y
    df[("correct", "")] = df.subject_choice.eq(df.optimal_action).all(axis=1)
    # get x value for correct action (handeling for multiple optimal actions)
    x_probs = df[x].values
    opt_mask = df.optimal_action.values.astype(bool)
    df[("x_prob", "value")] = (x_probs * opt_mask).sum(axis=1) / opt_mask.sum(axis=1)
    df[("x_prob", "bin")] = pd.cut(
        df[("x_prob", "value")],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    curve_df = df.groupby([("x_prob", "bin")], observed=True).correct.mean()
    # take mid of bin as output x value
    curve_df.index = curve_df.index.map(lambda x: x.mid)
    curve_df = curve_df.reset_index()
    curve_df.columns = [f"{x}_value", "p_correct"]
    return curve_df


def get_data():
    return gid.get_navigation_strategies_df(
        strategies=["structure", "habit"],
        n_history=1,
        verbose=False,
    )
