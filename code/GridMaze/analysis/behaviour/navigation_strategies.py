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


def get_navigation_strategy_weights(sessions, stim_on=False):
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
    result = minimize(get_neg_loglikelihood, initial_weights, args=(sessions, stim_on), method="BFGS")
    optimal_weights = result.x
    optimal_weight_vector, optimal_weight_structure, optimal_weight_penalty = optimal_weights
    return {
        "weight_vector": optimal_weight_vector,
        "weight_structure": optimal_weight_structure,
        "weight_penalty": optimal_weight_penalty,
    }


def get_neg_loglikelihood(weights, sessions, stim_on=False, remove_post_max_stim_dur=True):
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
    session_navigation_strategies_dfs = [session.navigation_strategies_df for session in sessions]
    navigation_strategies_df = pd.concat(session_navigation_strategies_dfs, axis=0, ignore_index=True)
    # remove redudant choices (given nan values in navigation_strateges df)
    navigation_strategies_df = navigation_strategies_df[~navigation_strategies_df.choice_value.isna().any(axis=1)]
    # select choices based on stim
    if stim_on is not None:
        navigation_strategies_df = navigation_strategies_df[navigation_strategies_df.stim_on == stim_on]
        # remove decisions after max stim duration to avoid bias between stim and non-stim trials
        if remove_post_max_stim_dur:
            navigation_strategies_df = navigation_strategies_df[
                navigation_strategies_df.time_in_trial.lt(MAX_STIM_DURATION)
            ]
    # get neg log likelihood
    V_vector = navigation_strategies_df.vector_navigation_value.to_numpy()
    V_structure = navigation_strategies_df.structure_navigation_value.to_numpy()
    V_penalty = navigation_strategies_df.penalty_value.to_numpy()
    A_bool = navigation_strategies_df.available.to_numpy()
    A = np.where(A_bool, 0, INVALID_TRANSITION)
    choice_mask = navigation_strategies_df.choice_value.to_numpy().astype(bool)
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


# %% Learning analysis


def plot_learning_msm_strategy_weights(days=[4, 5, 6, 7, 8], plot_penalty=True, axes=None):
    results = []
    for subject_info in SUBJECT_INFO.itertuples():
        sessions = gs.get_sessions(
            subject_IDs=[subject_info.subject_ID],
            experiment_phases="learning",
            learning_days=list(days),
            with_data=["navigation_strategies_df"],
        )
        strategy_weights = get_navigation_strategy_weights(sessions, stim_on=False)
        results.append(
            {
                "subject_ID": subject_info.subject_ID,
                "condition": subject_info.condition,
                "weight_vector": strategy_weights["weight_vector"],
                "weight_structure": strategy_weights["weight_structure"],
                "weight_penalty": strategy_weights["weight_penalty"],
            }
        )
    results_df = pd.DataFrame(results)
    # plotting
    n_vars = 3 if plot_penalty else 2
    if axes is None:

        f, axes = plt.subplots(1, n_vars, figsize=(2 * n_vars, 3), sharex=True)
        axes = axes.flatten()
    f.tight_layout()
    for var, ax, label in zip(
        ["weight_vector", "weight_structure", "weight_penalty"][:n_vars],
        axes,
        ["Vector", "Structure", "Penalty"][:n_vars],
    ):
        sns.pointplot(
            data=results_df,
            x="condition",
            y=var,
            ax=ax,
            hue="subject_ID",
            palette="tab10",
            alpha=1,
            dodge=0.2,
            legend=False,
            linestyles="-",
            errorbar=("ci", 95),
        )
        if var in ["weight_vector", "weight_structure"]:
            ax.set_ylim(-0.1, 1)
        # elif var == "weight_penalty":
        #     ax.set_ylim(-2, 0.1)
        ax.set_ylabel(label)
        ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


# %% Opto Analyses


def get_random_effects_strategy_weights(experiment_phase="learning", days="all"):
    """ """
    if experiment_phase == "learning":
        l_days = days
        s_days = []
    elif experiment_phase == "full_trial_stim":
        l_days = []
        s_days = days
    else:
        raise ValueError("experiment_phase must be 'learning' or 'full_trial_stim")
    results = []
    for subject_info in SUBJECT_INFO.itertuples():
        sessions = gs.get_sessions(
            subject_IDs=[subject_info.subject_ID],
            experiment_phases=experiment_phase,
            learning_days=l_days,
            stim_days=s_days,
            with_data=["navigation_strategies_df"],
        )
        for stim_on in [True, False]:
            strategy_weights = get_navigation_strategy_weights(sessions, stim_on=stim_on)
            results.append(
                {
                    "subject_ID": subject_info.subject_ID,
                    "sex": subject_info.sex,
                    "condition": subject_info.condition,
                    "stim_on": stim_on,
                    "weight_vector": strategy_weights["weight_vector"],
                    "weight_structure": strategy_weights["weight_structure"],
                    "weight_penalty": strategy_weights["weight_penalty"],
                }
            )
    return pd.DataFrame(results)


def plot_random_effects_results(results_df, colormap="viridis"):
    palette = sns.color_palette(colormap, 2)
    df_melted = pd.melt(
        results_df,
        id_vars=["condition", "stim_on"],
        value_vars=["weight_vector", "weight_structure"],
        var_name="measurement",
        value_name="value",
    )
    f, axes = plt.subplots(1, 2, figsize=(6, 3))
    f.tight_layout()
    for strat, ax in zip(["weight_vector", "weight_structure"], axes):
        legend = True if strat == "weight_structure" else False
        df = df_melted[df_melted["measurement"] == strat]
        sns.stripplot(
            data=df,
            x="condition",
            y="value",
            hue="stim_on",
            ax=ax,
            legend=legend,
            palette=palette,
            alpha=0.5,
            dodge=True,
        )
        sns.pointplot(
            data=df,
            x="condition",
            y="value",
            hue="stim_on",
            ax=ax,
            errorbar=("ci", 95),
            palette=palette,
            linestyles="",
            dodge=0.35,
            legend=False,
        )
        ax.set_ylabel(strat)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    return


def get_fixed_effects_strategy_weights(experiment_phase="learning", up_to_day=8):
    """ """
    results = []
    for day in np.arange(1, up_to_day + 1):
        for condition in ["opto", "control"]:
            sessions = gs.get_sessions(
                conditions=condition,
                experiment_phases=experiment_phase,
                learning_days=[day],
                with_data=["navigation_strategies_df"],
            )
            for stim_on in [True, False]:
                strategy_weights = get_navigation_strategy_weights(sessions, stim_on=stim_on)
                results.append(
                    {
                        "day": day,
                        "condition": condition,
                        "stim_on": stim_on,
                        "weight_vector": strategy_weights["weight_vector"],
                        "weight_structure": strategy_weights["weight_structure"],
                        "weight_penalty": strategy_weights["weight_penalty"],
                    }
                )
    return pd.DataFrame(results)


def plot_fixed_effects_timeseries(results_df):
    """ """
    line_styles = {
        ("opto", True): ("deepskyblue", "solid"),
        ("opto", False): ("deepskyblue", "dashed"),
        ("control", True): ("darkblue", "solid"),
        ("control", False): ("black", "dashed"),
    }
    fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    y_labels = ["Weight Vector", "Weight Structure"]
    columns = ["weight_vector", "weight_structure"]

    for i, ax in enumerate(axs):
        for (condition, stim_on), (color, style) in line_styles.items():
            legend = True if ax == axs[1] else False
            subset = results_df[(results_df["condition"] == condition) & (results_df["stim_on"] == stim_on)]
            sns.lineplot(
                x="day",
                y=columns[i],
                data=subset,
                ax=ax,
                label=f"{condition} stim_on={stim_on}",
                color=color,
                linestyle=style,
                legend=legend,
                alpha=0.7,
            )
            if legend:
                ax.legend(fontsize="small")

        ax.set_ylabel(y_labels[i])
        ax.set_xlabel("Day")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    return


# %% Within Subject Analysis


def get_within_subject_stim_comparisons(group="opto", plot_penalty=True, axes=None):
    """ """
    if group == "opto":
        subjects = SUBJECT_INFO[SUBJECT_INFO.condition == "opto"].subject_ID
    elif group == "control":
        subjects = ["mFC-opto_16"]
    results_df = []
    for subject in subjects:
        if SUBJECT_INFO[SUBJECT_INFO.subject_ID == subject].condition.values[0] != group:
            continue
        else:
            sessions = gs.get_sessions(subject_IDs=[subject], experiment_phases="full_trial_stim", ignore_sessions=True)
            if len(sessions) == 0:
                print(f"No stim sessions found for {subject}")
                continue

        for stim_on in [True, False]:
            strategy_weights = get_navigation_strategy_weights(sessions, stim_on=stim_on)
            results_df.append(
                {
                    "subject_ID": subject,
                    "stim_on": stim_on,
                    "weight_vector": strategy_weights["weight_vector"],
                    "weight_structure": strategy_weights["weight_structure"],
                    "weight_penalty": strategy_weights["weight_penalty"],
                }
            )
    results_df = pd.DataFrame(results_df)
    # compute delta weights
    stim_on_df = results_df[results_df.stim_on].set_index("subject_ID").drop(columns=["stim_on"])
    stim_off_df = results_df[~results_df.stim_on].set_index("subject_ID").drop(columns=["stim_on"])
    stim_diff_df = stim_on_df.subtract(stim_off_df)
    weight_stim_diff = stim_diff_df["weight_structure"].abs() - stim_diff_df["weight_vector"].abs()
    # plotting
    n_plt = 4 if plot_penalty else 3
    dodge = 0.1 if group == "opto" else False
    if axes is None:
        f, axes = plt.subplots(1, n_plt, figsize=(2 * n_plt, 3))
    axes = axes.flatten()
    for var, ax, label in zip(
        ["weight_vector", "weight_structure", "weight_penalty"][: n_plt - 1],
        axes,
        ["Vector", "Structure", "Penalty"][: n_plt - 1],
    ):
        # stats
        if not group == "control":
            t, p = ttest_rel(stim_on_df[var], stim_off_df[var])
            ax.set_title(f"T={t:.3f} p={p:.3f}", size=10)
        sns.pointplot(
            data=results_df,
            x="stim_on",
            y=var,
            ax=ax,
            hue="subject_ID",
            palette="tab10",
            alpha=1,
            legend=False,
            dodge=dodge,
        )
        if var in ["weight_vector", "weight_structure"]:
            ax.set_ylim(-0.1, 1)
        # elif var == "weight_penalty":
        #     ax.set_ylim(-2, 0.1)
        ax.set_ylabel(label)
        ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    if not group == "control":
        t_diff, p_diff = ttest_1samp(weight_stim_diff, 0)
        axes[-1].set_title(f"T={t_diff:.3f} p={p_diff:.3f}", size=10)
    sns.swarmplot(weight_stim_diff, ax=axes[-1], color="black", alpha=0.8, s=8)
    axes[-1].axhline(0, color="silver", linestyle="--", alpha=0.5)
    axes[-1].set_ylabel("Δstructure - Δvector")
    axes[-1].spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Subjects")
    f.tight_layout()


def get_within_subject_stim_comparisons2(group="opto"):
    """ """
    results_df = []
    if group == "opto":
        subjects = SUBJECT_INFO[SUBJECT_INFO.condition == "opto"].subject_ID
    elif group == "control":
        subjects = ["mFC-opto_16"]
    for subject in subjects:
        for stim_on in [True, False]:
            strategy_weights = get_navigation_strategy_weights2(subject, stim_on=stim_on)
            results_df.append(
                {
                    "subject_ID": subject,
                    "stim_on": stim_on,
                    "weight_vector": strategy_weights["weight_vector"],
                    "weight_structure": strategy_weights["weight_structure"],
                    "weight_habit": strategy_weights["weight_habit"],
                    "weight_penalty": strategy_weights["weight_penalty"],
                }
            )
    results_df = pd.DataFrame(results_df)
    # plotting
    f, axes = plt.subplots(1, 4, figsize=(9, 3))
    axes = axes.flatten()
    for var, ax in zip(["weight_vector", "weight_structure", "weight_habit", "weight_penalty"], axes):
        sns.pointplot(
            data=results_df, x="stim_on", y=var, ax=ax, hue="subject_ID", palette="dark", alpha=0.5, legend=False
        )
        ax.axhline(0, color="silver", linestyle="--", alpha=0.5)
    f.tight_layout()
    return results_df


# %% Fixed effect analysis per subject


def fixed_effect_msm_analysis(axes=None, plot_penalty=True):
    results_path = Path("../results/nsm_fixed_effects")
    valid_opto_subjects = SUBJECT_INFO[
        (SUBJECT_INFO.included_in_full_trial_stim) & (SUBJECT_INFO.condition == "opto")
    ].subject_ID.to_list()  # valid subjects in opto group
    valid_control_subjects = SUBJECT_INFO[
        (SUBJECT_INFO.included_in_full_trial_stim) & (SUBJECT_INFO.condition == "control")
    ].subject_ID.to_list()  # valid subjects in control group
    # check permutation have been run
    for valid_subjects in [valid_opto_subjects, valid_control_subjects]:
        for subject in valid_subjects:
            subject_results_path = results_path / subject
            if not (
                (subject_results_path / "null_distribution.tsv").exists()
                and (subject_results_path / "true_value.json").exists()
            ):
                raise FileExistsError(
                    f"Missing results data for {subject}, execute with 'weight_diff_fixed_effects_analysis' function."
                )
    # plot results
    subjects = valid_opto_subjects + valid_control_subjects + 6 * [None]  # add empty ax for spacing
    n_var = 3 if plot_penalty else 2
    if axes is None:
        f, axes = plt.subplots(7, 2 * n_var, figsize=(8, 8), clear=True, sharex=True, sharey=True)
    colors = sns.color_palette("tab10", 8)
    for i, subject in enumerate(subjects):
        row = i - 7 * (i // 7)
        if subject is None:  # leave empty ax where appropriate
            for j in range(n_var):
                column = j + n_var * (i // 7)
                ax = axes[row, column]
                ax.axis("off")
        else:
            null_distributions_df = pd.read_csv(results_path / subject / "null_distribution.tsv", sep="\t")
            with open(results_path / subject / "true_value.json", "r") as infile:
                true_values = json.load(infile)
            for j, (var, label) in enumerate(
                zip(
                    ["weight_vector_diff", "weight_structure_diff", "weight_penalty_diff"][:n_var],
                    ["Δ Vector", "Δ Structure", "Δ Penalty"][:n_var],
                )
            ):
                column = j + n_var * (i // 7)
                ax = axes[row, column]
                sns.histplot(
                    null_distributions_df,
                    x=var,
                    ax=ax,
                    bins=1000,
                    element="bars",
                    edgecolor=None,
                    alpha=0.8,
                    color=colors[i],
                )
                ax.axvline(true_values[var], color="black", linestyle="--")
                # get p-value
                x = null_distributions_df[var].gt(true_values[var]).mean()
                p = 2 * min(x, 1 - x)
                ax.set_title(f"{subject}: p={p:.3f}", size=8)
                ax.set_xlabel(var.split("_")[1])
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.set_xlabel(label)
    f.tight_layout()
    return


def weight_diff_fixed_effects_analysis(subject_ID, n=10_000, plot=True, return_data=False, save=False):
    sessions = gs.get_sessions(
        subject_IDs=[subject_ID], experiment_phases="full_trial_stim", with_data=["navigation_strategies_df"]
    )
    navigation_strategies_df = pd.concat(
        [session.navigation_strategies_df for session in sessions], axis=0, ignore_index=True
    )
    navigation_strategies_df = navigation_strategies_df[navigation_strategies_df.time_in_trial.lt(MAX_STIM_DURATION)]
    trial2stim = navigation_strategies_df[["trial", "stim_trial"]].drop_duplicates()
    # get true difference between weights
    true_results = []
    for stim in [False, True]:
        ns_stim_df = navigation_strategies_df[navigation_strategies_df.stim_on == stim]
        strategy_weights = get_navigation_strategy_weights3(ns_stim_df)
        true_results.append(
            {
                "stim_on": stim,
                "weight_vector": strategy_weights["weight_vector"],
                "weight_structure": strategy_weights["weight_structure"],
                "weight_penalty": strategy_weights["weight_penalty"],
            }
        )
    true_results_df = pd.DataFrame(true_results)
    true_diffs = {
        x + "_diff": true_results_df[x].diff().values[1]
        for x in ["weight_vector", "weight_structure", "weight_penalty"]
    }
    # get shuffled distributions
    stim_bool = trial2stim.stim_trial
    stim_permuations_df = pd.concat([stim_bool.sample(frac=1).reset_index(drop=True) for _ in range(n)], axis=1)
    stim_permuations_df.columns = np.arange(n)
    stim_permuations_df.index = trial2stim.trial
    shuffle_results = []
    for i in tqdm(range(n)):
        ns_df = navigation_strategies_df.copy()
        perm_stim_trial = ns_df.trial.map(stim_permuations_df[i].to_dict())  # shuffle stim trials
        ns_df["stim_on"] = perm_stim_trial
        for stim_on in [False, True]:
            ns_stim_df = ns_df[ns_df.stim_on == stim_on]
            strategy_weights = get_navigation_strategy_weights3(ns_stim_df)
            shuffle_results.append(
                {
                    "permutation": i,
                    "stim_on": stim_on,
                    "weight_vector": strategy_weights["weight_vector"],
                    "weight_structure": strategy_weights["weight_structure"],
                    "weight_penalty": strategy_weights["weight_penalty"],
                }
            )
    shuffle_results_df = pd.DataFrame(shuffle_results)
    # calculate differences between stim on and off for each strategy weight
    pivoted_df = shuffle_results_df.pivot(
        index="permutation", columns="stim_on", values=["weight_vector", "weight_structure", "weight_penalty"]
    )
    diff_df = pivoted_df.xs(True, axis=1, level=1) - pivoted_df.xs(False, axis=1, level=1)
    diff_df.columns = ["weight_vector_diff", "weight_structure_diff", "weight_penalty_diff"]
    # plotting
    if plot:
        f, axes = plt.subplots(1, 3, figsize=(10, 3))
        for var, ax in zip(["weight_vector_diff", "weight_structure_diff", "weight_penalty_diff"], axes.flatten()):
            sns.histplot(diff_df, x=var, ax=ax, bins=100, element="step", alpha=0.5, color="black")
            ax.axvline(true_diffs[var], color="red")
            # get p-value
            x = diff_df[var].gt(true_diffs[var]).mean()
            p = 2 * min(x, 1 - x)
            ax.set_title(f"p={p:.3f}")
    if save:
        results_path = Path("../results/nsm_fixed_effects")
        subject_results_path = results_path / subject_ID
        if not subject_results_path.exists():
            subject_results_path.mkdir(parents=True)
        # save permuted null distribution
        diff_df.to_csv(subject_results_path / "null_distribution.tsv", sep="\t", index=False)
        # save true value
        with open(subject_results_path / "true_value.json", "w") as outfile:
            outfile.write(json.dumps(true_diffs, indent=4))
    if return_data:
        return pd.DataFrame(true_diffs), diff_df


def get_navigation_strategy_weights3(navigation_strategies_df):
    """ """
    initial_weights = [0, 0, 0]
    result = minimize(get_neg_loglikelihood3, initial_weights, args=(navigation_strategies_df), method="BFGS")
    optimal_weights = result.x
    optimal_weight_vector, optimal_weight_structure, optimal_weight_penalty = optimal_weights
    return {
        "weight_vector": optimal_weight_vector,
        "weight_structure": optimal_weight_structure,
        "weight_penalty": optimal_weight_penalty,
    }


def get_neg_loglikelihood3(weights, navigation_strategies_df):
    """ """
    weight_vector, weight_structure, weight_penalty = weights
    # remove redudant choices (given nan values in navigation_strateges df)
    navigation_strategies_df = navigation_strategies_df[~navigation_strategies_df.choice_value.isna().any(axis=1)]
    # get neg log likelihood
    V_vector = navigation_strategies_df.vector_navigation_value.to_numpy()
    V_structure = navigation_strategies_df.structure_navigation_value.to_numpy()
    V_penalty = navigation_strategies_df.penalty_value.to_numpy()
    A_bool = navigation_strategies_df.available.to_numpy()
    A = np.where(A_bool, 0, INVALID_TRANSITION)
    choice_mask = navigation_strategies_df.choice_value.to_numpy().astype(bool)
    V = weight_vector * V_vector + weight_structure * V_structure + weight_penalty * V_penalty + A
    P = softmax(V, choice_mask)
    loglikelihood = np.log(P)
    if np.any(np.isnan(loglikelihood)):
        assert ValueError("Log likelihood contains NaN(s).")
    return -np.sum(np.log(P))


# %%


def fixed_effects_pvp_plot(axes=None, plot_penalty=True):
    results_path = Path("../results/nsm_fixed_effects")
    n_var = 3 if plot_penalty else 2
    if axes is None:
        f, axes = plt.subplots(1, n_var, figsize=(3 * n_var, 3), sharex=True, sharey=True)
    valid_subjects = SUBJECT_INFO[
        (SUBJECT_INFO.included_in_full_trial_stim) & (SUBJECT_INFO.condition == "opto")
    ].subject_ID.to_list()
    p_values = []
    for subject in valid_subjects:
        # loda data
        null_distributions_df = pd.read_csv(results_path / subject / "null_distribution.tsv", sep="\t")
        with open(results_path / subject / "true_value.json", "r") as infile:
            true_values = json.load(infile)
        # compute p values
        for var, label in zip(
            ["weight_vector_diff", "weight_structure_diff", "weight_penalty_diff"][:n_var],
            ["Δ Vector", "Δ Structure", "Δ Penalty"][:n_var],
        ):
            x = null_distributions_df[var].gt(true_values[var]).mean()
            p = 2 * min(x, 1 - x)
            p_values.append({"subject_ID": subject, "variable": var, "p_value": p})
    # plotting
    p_values_df = pd.DataFrame(p_values)
    for i, var in enumerate(["weight_vector_diff", "weight_structure_diff", "weight_penalty_diff"][:n_var]):
        ax = axes[i]
        sorted_p = sorted(p_values_df[p_values_df.variable == var].p_value.to_list())
        ax.plot([0, len(sorted_p)], [0, 1], color="black", linestyle="--", alpha=0.5)
        ax.scatter(
            np.arange(len(sorted_p)),
            sorted_p,
            color="purple",
            alpha=1,
        )
        ax.set_title(var.split("_")[1])
        ax.set_xlabel("Ordered p-values")
        ax.set_ylabel("Expected p-value")
        ax.set_ylim(-0.1, 1)
        ax.set_xticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
