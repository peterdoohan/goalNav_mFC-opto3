"""
how quickly to subjects correct their behaviour after an error,
e.g. looking at backtracking after an error
"""

# %% Imports
import json
import numpy as np
import pandas as pd

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.errors import errors as err

# %% Global Variables
MAX_STIM_DURATION = 30

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Functions


def get_error_centered_df(
    error_df,
    n=3,
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    outlier_thres=None,
    stim_only=True,
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
    dfs = []
    for t in trial_unique_IDs:
        trial_df = df[df.trial_unique_ID == t]
        error_inds = trial_df.index[trial_df.error].tolist()
        if len(error_inds) == 0:
            continue
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
        dfs.append(err_cent_df.reset_index(drop=True))

    output_df = pd.concat(dfs, ignore_index=True)
    return output_df


def get_prop_backtrack_after_error(df, n=3):
    """ """
    err_offsets = list(range(-n, 0)) + [0] + list(range(1, n + 1))
    trial_unique_IDs = df.trial_unique_ID.unique()
    dfs = []
    for t in trial_unique_IDs:
        trial_df = df[df.trial_unique_ID == t]
        error_inds = trial_df.index[trial_df.error].tolist()
        if len(error_inds) == 0:
            continue
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
        dfs.append(err_cent_df.reset_index(drop=True))

    return pd.concat(dfs, ignore_index=True)
