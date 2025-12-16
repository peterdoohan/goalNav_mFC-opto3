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
# %%

# %%


# %% Modelling functions


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
            strategy_weights = get_navigation_strategy_weights(_df)
            strategy_weights["subject_ID"] = subject
            strategy_weights["condition"] = SUBJECT_INFO.set_index("subject_ID").loc[subject].condition
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
