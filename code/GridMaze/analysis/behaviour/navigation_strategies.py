"""
This moduel models choices during maze navigation as a function of vector based and shortest path based strategies.
And visualizes the results.
"""

# %% Imports
import sys
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import date
import pandas as pd
from scipy.optimize import minimize
from GridMaze.analysis.core import get_sessions as gs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_1samp

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# %% Global variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

INVALID_TRANSITION = -100
LOG_MAX_FLOAT = np.log(sys.float_info.max / 2.1)


with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

SUBJECT_INFO = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

MAX_STIM_DURATION = 30  # seconds
# %%

# %%


# %% Modelling functions


def get_random_effects_summary_df(stim_on_only=False, max_trial_duration=120):
    """ """
    input_data = get_input_data()
    results = []
    for subject in SUBJECT_IDS:
        df = input_data[input_data.subject_ID == subject]
        # filter data
        if max_trial_duration is not None:
            df = df[df.trial_duration.le(max_trial_duration)]
        for stim_trial in [True, False]:
            _df = df[df.stim_trial == stim_trial]
            if stim_on_only:
                if stim_trial:
                    _df = _df[_df.stim_on]
                else:
                    # must match statistics between conditions when filtering for stim_on
                    _df = _df[_df.time_in_trial.le(MAX_STIM_DURATION)]
            # fit strategy weights on select data
            strategy_weights = get_navigation_strategy_weights(_df)
            strategy_weights["subject_ID"] = subject
            strategy_weights["condition"] = SUBJECT_INFO.set_index("subject_ID").loc[subject].condition
            strategy_weights["stim_on"] = stim_trial
            results.append(strategy_weights)

    return pd.DataFrame(results)


def get_input_data(
    subject_IDs="all",
    conditions="all",
    experiment_phases="expert",
    total_stim_days="all",
):
    """ """
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
        dfs.append(nav_strats_df)
    input_df = pd.concat(dfs, axis=0, ignore_index=True)
    return input_df


def get_navigation_strategy_weights(df):
    """
    Calculates the optimal weights for the vector navigation weight, structure navigation weight, and penalty weight using maximum likelihood estimation.

    Parameters
    ----------
    sessions : list of Session
        A list of Session objects for which to calculate the optimal weights.

    Returns
    -------
    dict
        A dictionary containing the optimal weights for the vector navigation value, structure navigation value, and penalty value.
        The keys are 'weight_vector', 'weight_structure', and 'weight_penalty', respectively.
    """
    initial_weights = [0, 0, 0]
    result = minimize(get_neg_loglikelihood, initial_weights, args=(df,), method="BFGS")
    optimal_weights = result.x
    optimal_weight_vector, optimal_weight_structure, optimal_weight_penalty = optimal_weights
    return {
        "weight_vector": optimal_weight_vector,
        "weight_structure": optimal_weight_structure,
        "weight_penalty": optimal_weight_penalty,
    }


def get_neg_loglikelihood(df, weights):
    """
    Calculates the negative log likelihood of the data given the vector_navigation, structure_navigation and penalty_weights.

    Parameters
    ----------
    weights : tuple
        A tuple of three floats representing the weights for the vector navigation value, structure navigation value, and penalty value, respectively.
    sessions : list of Session
        A list of Session objects for which to calculate the negative log likelihood.
    stim_on : bool, to include choices where opto stim was on or off (False)

    Returns
    -------
    float
        The negative log likelihood of the data given the weights.
    """
    weight_vector, weight_structure, weight_penalty = weights
    # get neg log likelihood
    V_vector = df.vector_navigation_value.to_numpy()
    V_structure = df.structure_navigation_value.to_numpy()
    V_penalty = df.penalty_value.to_numpy()
    A_bool = df.available.to_numpy()
    A = np.where(A_bool, 0, INVALID_TRANSITION)
    choice_mask = df.choice_value.to_numpy().astype(bool)
    V = weight_vector * V_vector + weight_structure * V_structure + weight_penalty * V_penalty + A
    P = softmax(V, choice_mask)
    loglikelihood = np.log(P)
    if np.any(np.isnan(loglikelihood)):
        assert ValueError("Log likelihood contains NaN(s).")
    return -np.sum(np.log(P))


def softmax(V, choice_mask):
    """Calculates softmax probabilities for choices in a given state."""
    V[V > LOG_MAX_FLOAT] = LOG_MAX_FLOAT  # Protection against overflow in exponential.
    expV = np.exp(V)
    return expV[choice_mask] / np.sum(expV, axis=1)
