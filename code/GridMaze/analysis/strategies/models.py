"""
This moduel models choices during maze navigation as a function of vector based and shortest path based strategies.
And visualizes the results.
"""

# %% Imports
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from GridMaze.analysis.core import get_sessions as gs
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import mixed_anova


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

# %% Modelling functions


def get_navigation_strategy_weights(df):
    """
    Calculates the optimal weights for the vector navigation weight, structure navigation weight, and penalty weight using maximum likelihood estimation.

    Parameters
    ----------
    df : pd.DataFrame

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


def get_neg_loglikelihood(weights, df):
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
