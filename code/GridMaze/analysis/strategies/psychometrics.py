"""
Make some psycometric curves for P(correct) vs Habit Values
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pingouin import mixed_anova
from scipy.optimize import curve_fit


from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.strategies import get_input_data as gid


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %% Functions


def get_curve_fit_df(psy_curve_df):
    fit_results = []
    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            sub_df = psy_curve_df[(psy_curve_df.subject_ID == subject) & (psy_curve_df.stim_trial == stim_trial)]
            x = sub_df.habit_value.values.astype(float)
            y = sub_df.p_correct.values.astype(float)
            fit_params = get_sigmoid_curve(x, y, return_as="params")
            fit_params["subject_ID"] = subject
            fit_params["stim_trial"] = stim_trial
            fit_params["condition"] = sub_df.condition.unique()[0]
            fit_results.append(fit_params)
    fit_df = pd.DataFrame(fit_results)
    return fit_df


def plot_habit_psychometrics_inset(
    psy_curve_df,
    group="opto",
    habit_value_range=(0, 0.25),
    stim_color="#0077FF",
    ax=None,
):
    # set up fig
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(0.75, 1.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    df = psy_curve_df.copy()
    df["habit_value"] = df.habit_value.astype(float)
    df = df[df.habit_value.between(*habit_value_range)]
    grouped_df = df.groupby(["condition", "stim_trial", "habit_value"], observed=True).p_correct
    mean = grouped_df.mean()
    sem = grouped_df.sem()

    # plot
    for stim_trial, color in zip([False, True], ["dimgrey", stim_color]):
        # plot error bars
        _mean = mean.loc[(group, stim_trial)]
        _sem = sem.loc[(group, stim_trial)]
        ax.errorbar(
            _mean.index,
            _mean.values,
            yerr=_sem.values,
            fmt="o",
            label="light on" if stim_trial else "light off",
            color=color,
            markersize=4,
            alpha=1,
        )
        ax.set_title(group)
    ax.set_xlim(*habit_value_range)

    return


def plot_habit_psychometrics_summary(
    psy_curve_df,
    stim_color="#0077FF",
    print_stats=True,
    axes=None,
):
    """ """
    # set up fig
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
            # plot error bars
            _mean = mean.loc[(cond, stim_trial)]
            _sem = sem.loc[(cond, stim_trial)]
            ax.errorbar(
                _mean.index,
                _mean.values,
                yerr=_sem.values,
                fmt="o",
                label="light on" if stim_trial else "light off",
                color=color,
                markersize=4,
                alpha=1,
            )
            # plot fitted sigmoid curve
            x, y = _mean.index.values.astype(float), _mean.values.astype(float)
            x_fit, y_fit = get_sigmoid_curve(x, y)
            ax.plot(x_fit, y_fit, color=color)
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
    stim_day_range=(4, gs.TOTAL_STIM_DAYS),
    stim_only=True,
    decision_points_only=False,
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
    if decision_points_only:
        # nodes with degree 3 or 4
        df = df[df.available.sum(axis=1).gt(2)]

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


# %% sigmoid curve fitting


def get_sigmoid_curve(x, y, return_as="line"):
    p0 = [0, 10.0, -100, 0.05]  # alpha  # beta  # gamma  # lambda
    bounds = ([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf])
    # curve fit
    popt, pcov = curve_fit(sigmoid_4p, x, y, p0=p0, bounds=bounds, maxfev=100_000)
    # generate fit line
    x_fit = np.linspace(0, 1, 500)
    y_fit = sigmoid_4p(x_fit, *popt)
    if return_as == "line":
        return x_fit, y_fit
    elif return_as == "params":
        return {"alpha": popt[0], "beta": popt[1], "gamma": popt[2], "lambda": popt[3]}
    else:
        raise ValueError("return_as must be 'line' or 'params'")


def sigmoid_4p(x, alpha, beta, gamma, lam):
    """
    alpha : threshold (inflection point)
    beta  : slope
    gamma : lower asymptote (guess rate)
    lam   : lapse rate (upper asymptote = 1 - lam)
    """
    return gamma + (1 - gamma - lam) / (1 + np.exp(-beta * (x - alpha)))
