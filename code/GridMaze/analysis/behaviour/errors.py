"""
Quant of errors during navigation
"""

# %% Imports
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from GridMaze.analysis.core import get_sessions as gs


# %% Global Variables


# %% Functions


def plot_trial_distance_to_goal(session, trial=2, ax=None):
    """ """
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    cue_time = trials_df.loc[trial, ("time", "cue")]
    # filter for trial data
    df = session.trajectory_decisions_df.copy()
    trial_df = df[(df.trial == trial) & (df.trial_phase == "navigation")]

    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("steps-to-goal")
    ax.set_title(f"trial {trial}")

    # plot steps over time
    steps = trial_df.steps_to_goal.values
    time = trial_df.time.values
    time = time - cue_time  # align to trial start
    stim_on = trial_df.stim_on.values
    if np.any(stim_on):
        ax.plot(time[stim_on], steps[stim_on], color="#0077FF", lw=6, alpha=0.5)
    ax.plot(time, steps, color="black", lw=1)
    ax.scatter(time, steps, color="black", s=0.75)
    # plot errors
    error_mask = get_error_mask(trial_df.steps_to_goal)
    if np.any(error_mask):
        etime = time[error_mask]
        y = np.clip(steps[error_mask] - 1.5, a_min=0, a_max=None)
        ax.scatter(etime, y, color="red", marker="x")

    return


def get_error_times(session, trial, rel=True):
    """
    convience function
    """
    # filter for trial data
    df = session.trajectory_decisions_df.copy()
    trial_df = df[(df.trial == trial) & (df.trial_phase == "navigation")]
    error_mask = get_error_mask(trial_df.steps_to_goal)
    if error_mask.sum() == 0:
        return None
    error_times = trial_df.time.values[error_mask]
    if rel:
        # align times to trial start
        trials_df = session.trials_df.copy()
        trials_df.set_index("trial", inplace=True)
        cue_time = trials_df.loc[trial, ("time", "cue")]
        error_times = error_times - cue_time
    return error_times


def get_error_mask(steps, n=2):
    """
    True only at the first increase after n consecutive decreases.
    """
    diffs = steps.diff()
    increase = diffs > 0  #
    # previous n diffs were all negative
    prev_n_decreasing = diffs.shift(1).rolling(n, min_periods=n).max() < 0
    # detect is True at the first increasing index i; shift it back to i-1
    error_mask = (increase & prev_n_decreasing).fillna(False)
    error_mask = error_mask.shift(-1)
    error_mask.iloc[-1] = False  # avoid NaN at end
    return error_mask.values.astype(bool)
