"""
how quickly to subjects correct their behaviour after an error,
e.g. looking at backtracking after an error
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.errors import errors as err

# %% Global Variables
MAX_STIM_DURATION = 30

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Functions


def plot_group_by_stim_error_backtrack(
    results_df,
    n=[2, 3],
    ax=None,
    stim_color="#0077FF",
    print_stats=True,
):
    """ """
    df = results_df.copy()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))

    # average to get prob bracktracking across subjects
    subj_avg = df.groupby(["condition", "subject_ID", "stim_trial"]).error_offset.mean().error_offset
    # select err offset
    if isinstance(n, int):
        offset_df = subj_avg[n].reset_index(name="prob_backtrack")
    elif isinstance(n, list):
        offset_df = subj_avg[n].mean(axis=1).reset_index(name="prob_backtrack")
    else:
        raise ValueError("n must be int or list of ints")

    # plot
    cp.plot_group_by_stim(
        offset_df,
        y="prob_backtrack",
        ax=ax,
        stim_color=stim_color,
        print_stats=print_stats,
    )


def plot_prob_backtrack_after_error(
    results_df,
    stim_color="#0077FF",
    axes=None,
    x_range=None,
):
    """ """
    df = results_df.copy()

    # set up figure
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True, sharey=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("steps since error")
        ax.set_ylabel("P(backtrack)")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    # average to get prob bracktracking across subjects
    subj_avg = df.groupby(["condition", "subject_ID", "stim_trial"]).error_offset.mean()
    cond_grouped = subj_avg.groupby(level=[0, 2])
    mean = cond_grouped.mean().error_offset
    sem = cond_grouped.sem().error_offset

    # plotting
    for group, ax in zip(["control", "opto"], axes):
        for stim_trial, color in zip([False, True], ["grey", stim_color]):
            _mean = mean.loc[group, stim_trial]
            _sem = sem.loc[group, stim_trial]
            x = _mean.index.values.astype(int)
            y = _mean.values.astype(float)
            yerr = _sem.values.astype(float)
            ax.plot(x, y, label=f"stim={stim_trial}", color=color)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.25)
        if x_range is not None:
            ax.set_xlim(x_range)
        ax.set_ylim(top=0.2)
    axes[0].legend(fontsize="x-small", loc="upper left")


def get_error_centered_df(
    error_df,
    n=5,
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    stim_only=True,
    n_jobs=-1,
):
    df = err._filter_error_df(
        error_df,
        e="error",
        stim_day_range=stim_day_range,
        outlier_thres=outlier_thres,
        stim_only=stim_only,
    )

    err_offsets = list(range(-n, 0)) + [0] + list(range(1, n + 1))
    trial_unique_IDs = df.trial_unique_ID.unique()

    def _process_trial(trial_df, t):
        trial_df = df[df.trial_unique_ID == t]
        error_inds = trial_df.index[trial_df.error].tolist()
        if len(error_inds) == 0:
            return None
        _df = pd.DataFrame(index=error_inds, columns=err_offsets, data=np.nan)
        for err_ind in error_inds:
            for e_off in err_offsets:
                step_ind = err_ind + e_off
                if step_ind in trial_df.index:
                    _df.loc[err_ind, e_off] = int(trial_df.loc[step_ind, "backtracking_mask"])
                else:
                    continue  # nan by defualt
        err_cent_df = _df
        err_cent_df.columns = pd.MultiIndex.from_product([["error_offset"], err_cent_df.columns])
        err_cent_df[("trial_unique_ID", "")] = t
        err_cent_df[("error_index", "")] = error_inds
        err_cent_df[("condition", "")] = trial_df.condition.unique()[0]
        err_cent_df[("subject_ID", "")] = trial_df.subject_ID.unique()[0]
        err_cent_df[("stim_trial", "")] = trial_df.stim_trial.unique()[0]
        return err_cent_df.reset_index(drop=True)

    if n_jobs:
        dfs = Parallel(n_jobs=n_jobs)(delayed(_process_trial)(df, t) for t in trial_unique_IDs)
    else:
        dfs = [_process_trial(df, t) for t in trial_unique_IDs]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True)
