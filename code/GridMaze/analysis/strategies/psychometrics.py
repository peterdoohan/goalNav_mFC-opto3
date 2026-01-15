"""
Make some psycometric curves for P(correct) vs Habit Values
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pingouin import mixed_anova


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.strategies import get_input_data as gid


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %% Functions


def plot_habit_psychometrics_summary(psy_curve_df, stim_color="#0077FF", print_stats=True, axes=None):
    """ """
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2.5), sharey=True, sharex=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("correct \n habit value")
        ax.axvline(0.5, color="k", ls="--", alpha=0.5)
    axes[0].set_ylabel("P(correct)")

    grouped_df = psy_curve_df.groupby(["condition", "stim_trial", "habit_value"], observed=True).p_correct
    mean = grouped_df.mean()
    sem = grouped_df.sem()
    for ax, cond in zip(axes, ["control", "opto"]):
        for stim_trial, color in zip([False, True], ["dimgrey", stim_color]):
            _mean = mean.loc[(cond, stim_trial)]
            _sem = sem.loc[(cond, stim_trial)]
            ax.errorbar(
                _mean.index,
                _mean.values,
                yerr=_sem.values,
                label="light on" if stim_trial else "light off",
                color=color,
                marker="o",
                markersize=5,
                linestyle="-",
                alpha=0.8,
            )
            ax.set_title(cond)
        ax.set_ylim(0, 1.1)

    axes[0].legend(fontsize="x-small")

    if print_stats:
        for hv in psy_curve_df.habit_value.unique():
            stats_df = mixed_anova(
                dv="p_correct",
                within="stim_trial",
                between="condition",
                subject="subject_ID",
                data=psy_curve_df[psy_curve_df.habit_value == hv],
            )
            p_int = stats_df.loc[stats_df["Source"] == "Interaction"].iloc[0]
            print(f"Habit value {hv:.2f}: group x stim p={p_int['p-unc']:.3f}")


def get_psychometrics_df(
    navigation_strategies_df,
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    x="habit",
    x_bins=8,
):
    """ """
    # filter data
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]

    dfs = []
    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            subj_df = df[(df.subject_ID == subject) & (df.stim_trial == stim_trial)].copy()
            psy_curve = get_psychometric_curve(subj_df, x=x, bins=x_bins)
            psy_curve["subject_ID"] = subject
            psy_curve["condition"] = subj_df.condition.unique()[0]
            psy_curve["stim_trial"] = stim_trial
            dfs.append(psy_curve)

    psy_curve_df = pd.concat(dfs, ignore_index=True)
    return psy_curve_df


def get_psychometric_curve(df, x="habit", bins=5):
    """ """
    # add column for if subject choice == y
    df[("correct", "")] = ((df.subject_choice == 1) & (df.optimal_action == 1)).any(axis=1)
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
