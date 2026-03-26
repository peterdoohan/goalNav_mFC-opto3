""" """

# %% Imports
import json
from matplotlib import lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import permute

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.strategies import models
from GridMaze.analysis.strategies import get_input_data as gid

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds


# %% Other types of stats


def permute_weights(results_df, strategy="structure", n=5_000, sign="lower"):
    """ """
    df = results_df.copy()

    # get subject group info
    true_subject2condition = (
        df[["subject_ID", "condition"]].drop_duplicates().set_index("subject_ID").condition.to_dict()
    )
    all_subjects = df.subject_ID.unique()
    opto_subjects = [s for s, c in true_subject2condition.items() if c == "opto"]
    control_subjects = [s for s, c in true_subject2condition.items() if c == "control"]
    n_opto = len(opto_subjects)
    n_control = len(control_subjects)

    # get difference in strat weights between stim on and off across subejcts
    pivot_df = df.pivot(index="subject_ID", columns="stim_trial", values=strategy)
    subject_diffs = pivot_df.diff(axis=1)[True]

    true_group_diff = subject_diffs.loc[opto_subjects].mean() - subject_diffs.loc[control_subjects].mean()

    permuted_diffs = []
    for _ in range(n):
        np.random.shuffle(all_subjects)
        permuted_opto = all_subjects[:n_opto]
        permuted_control = all_subjects[n_opto : n_opto + n_control]
        permuted_diff = subject_diffs.loc[permuted_opto].mean() - subject_diffs.loc[permuted_control].mean()
        permuted_diffs.append(permuted_diff)

    # get p-value
    permuted_diffs = np.array(permuted_diffs)
    if sign == "lower":
        p_value = np.mean(permuted_diffs <= true_group_diff)
    elif sign == "upper":
        p_value = np.mean(permuted_diffs >= true_group_diff)
    elif sign is None:
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(true_group_diff))
    else:
        raise ValueError("sign must be 'lower', 'upper' or None")
    return true_group_diff, permuted_diffs, p_value


# %%


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
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    steps_to_goal_range=None,
    max_trial_duration=None,
    stim_only=False,
    subsample_non_stim_trials=False,
    decision_point_only=False,
    vector_structure_different=False,
    goal_sight_only=False,
    ignore_sessions_with_issues=False,
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
        # to check for sampling bias effects (seems fine)
        stim_trials = df[df.stim_trial].trial_unique_ID.to_list()
        keep_non_stim_trials = _subsample_non_stim_trials(df)
        df = df[df.trial_unique_ID.isin(stim_trials + keep_non_stim_trials)]
    if goal_sight_only:
        df = df[df.goal_sight]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]
    if steps_to_goal_range is not None:
        df = df[df.steps_to_goal.between(*steps_to_goal_range)]
    if decision_point_only:
        df = df[df.node_degree.gt(2)]
    # vector and structure strategies disagree
    if vector_structure_different:
        vec = df.vector.copy()
        vector_choice = vec.idxmax(axis=1)
        struc_bool_df = df.structure.eq(1)
        col_positions = struc_bool_df.columns.get_indexer(vector_choice)
        arr = struc_bool_df.to_numpy(dtype=bool)
        mask = arr[np.arange(len(struc_bool_df)), col_positions]
        df = df[mask]
    if ignore_sessions_with_issues:
        df = df[~df.noted_session_issues]

    # fit nav strategy weights for stim_on and stim_off decisions per subject
    results = []
    for subject in SUBJECT_IDS:
        subj_df = df[df.subject_ID == subject]
        condition = subj_df.condition.unique()[0]
        for stim_trial in [True, False]:
            _df = subj_df[subj_df.stim_trial == stim_trial]
            # fit strategy weights on select data
            strategy_weights = models.get_navigation_strategy_weights(_df, strategies=strategies, zscore=True)
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


# %% see if interaction terms acutally improve model fits relative to added params


def plot_habit_interaction_comparison(NLL_df, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("habit interactions")
    ax.set_ylabel("neg loglikelihood \n (z-scored)")

    # zscore within subject
    NLL_df["NLL_z"] = NLL_df.groupby("subject_ID")["NLL"].transform(lambda x: (x - x.mean()) / x.std())

    # plot
    sns.pointplot(
        data=NLL_df,
        x="habit_int",
        order=["none", "structure", "vector"],
        y="NLL_z",
        hue="condition",
        palette=["dimgray", "#0077FF"],
        errorbar="se",
        linestyle="none",
        alpha=1,
        dodge=0.3,
        ax=ax,
    )
    ax.legend(fontsize="small")


def get_habit_interaction_NLL_df():
    df = gid.get_navigation_strategies_df(
        strategies=[
            "vector",
            "structure",
            "habit",
            "structure_X_habit",
            "vector_X_habit",
            "backtracking_penalty",
        ],
    )
    # remove stim trials
    df = df[~df.stim_trial]

    results = []
    for subject in SUBJECT_IDS:
        subj_df = df[df.subject_ID == subject]
        condition = subj_df.condition.unique()[0]
        # fit strategy weights on select data
        for strategies, int_var in zip(
            [
                ["vector", "structure", "habit", "backtracking_penalty"],
                ["vector", "structure", "habit", "structure_X_habit", "backtracking_penalty"],
                ["vector", "structure", "habit", "vector_X_habit", "backtracking_penalty"],
            ],
            ["none", "structure", "vector"],
        ):
            strategy_weights = models.get_navigation_strategy_weights(
                subj_df,
                strategies=strategies,
            )
            # get best neg loglikelihood
            NLL = models.get_neg_loglikelihood(
                list(strategy_weights.values()),
                strategies,
                subj_df,
            )
            results.append(
                {
                    "subject_ID": subject,
                    "condition": condition,
                    "habit_int": int_var,
                    "NLL": NLL,
                }
            )
    return pd.DataFrame(results)


# %% compare optimal history lengths


def plot_history_length_comparison(NLL_df, ax=None):
    """ """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("habit history \n length")
    ax.set_ylabel("neg loglikelihood \n (z-scored)")

    # zscore within subject
    NLL_df["NLL_z"] = NLL_df.groupby("subject_ID")["NLL"].transform(lambda x: (x - x.mean()) / x.std())
    sns.lineplot(
        data=NLL_df,
        x="n_history",
        y="NLL_z",
        hue="condition",
        palette=["dimgray", "#0077FF"],
        errorbar="se",
        ax=ax,
    )
    ax.legend(fontsize="small")
    h = NLL_df.n_history.unique()
    ax.set_xticks(h)
    ax.set_xticklabels(h)
    return


def compare_habit_history_lengths(
    strategies=["vector", "structure", "habit", "backtracking_penalty"],
    n_range=(0, 3),
    sessions=None,
    verbose=True,
    save=False,
):
    """ """
    save_path = RESULTS_PATH / "strategies" / "habit_history_length_comparison.htsv"
    if not save and save_path.exists():
        if verbose:
            print(f"loading saved results from {save_path}")
        NLL_df = pd.read_csv(save_path, sep="\t")
        return NLL_df
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            stim_only=True,
            with_data=["navigation_df", "trials_df", "session_info"],
            must_have_data=True,
        )
    results = []
    for n in range(n_range[0], n_range[1] + 1):
        if verbose:
            print(f"n_history = {n}")
            print(f"generating input data...")
        nav_strat_df = gid.get_navigation_strategies_df(
            strategies=strategies,
            n_history=n,
            sessions=sessions,
            verbose=False,
            n_jobs=-1,
        )
        # filter for non-stim trials only (best fit normal data)
        df = nav_strat_df[~nav_strat_df.stim_trial]
        if verbose:
            print(f"fitting models...")
        for subject in SUBJECT_IDS:
            if verbose:
                print(f"  subject: {subject}")
            subj_df = df[df.subject_ID == subject]
            condition = subj_df.condition.unique()[0]
            # fit strategy weights on select data
            strategy_weights = models.get_navigation_strategy_weights(
                subj_df,
                strategies=strategies,
            )
            # get best neg loglikelihood
            NLL = models.get_neg_loglikelihood(
                list(strategy_weights.values()),
                strategies,
                subj_df,
            )
            results.append(
                {
                    "subject_ID": subject,
                    "condition": condition,
                    "n_history": n,
                    "NLL": NLL,
                }
            )
    NLL_df = pd.DataFrame(results)
    if save:
        if verbose:
            print(f"saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        NLL_df.to_csv(save_path, sep="\t", index=False)
    return NLL_df
