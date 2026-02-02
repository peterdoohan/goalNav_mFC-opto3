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
from GridMaze.analysis.core import plotting as cp
from GridMaze.analysis.strategies import get_input_data as gid


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

MAX_STIM_DURATION = 30  # seconds

# %%


def get_prop_optimal_df(
    navigation_strategies_df,
    strat_1="vector",
    strat_2="structure",
    stim_day_range=(6, gs.TOTAL_STIM_DAYS),
    stim_only=False,
    decision_points_only=False,
    ax=None,
):
    """ """
    df = navigation_strategies_df.copy()
    if stim_day_range is not None:
        df = df[df.total_stim_days.between(*stim_day_range)]
    if stim_only:
        df = df[df.time_in_trial.le(MAX_STIM_DURATION)]
    if decision_points_only:
        # nodes with degree 3 or 4
        df = df[df.available.sum(axis=1).gt(2)]
    # filter for points where strats disagree
    opt_mask = df.optimal_action.values.astype(bool)
    (x_probs * opt_mask).sum(axis=1) / opt_mask.sum(axis=1)

    v1 = df[strat_1].copy()
    v1[df.available] = -1
    v2 = df[strat_2].copy()
    v2[df.available] = -1
    v1_choice = v1.eq(df.idxmax(axis=1))
    v2_choice = v2.eq(1)
    col_positions = v2_choice.columns.get_indexer(v1_choice)
    arr = v2_choice.to_numpy(dtype=bool)
    mask = arr[np.arange(len(v2_choice)), col_positions]
    df = df[mask]
    # on filtered data not if subjects were correct
    df[("correct", "")] = (df.subject_choice.eq(df.optimal_action) & df.subject_choice.eq(1)).any(axis=1)

    res = df.groupby(["subject_ID", "condition", "stim_trial"]).correct.mean().reset_index(name="correct")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    cp.plot_group_by_stim(res, y="correct", print_stats=True, ax=ax)
    ax.set_ylim(0.60, 0.9)


# %% Functions


def plot_sigmoid_fit_params(fit_df, p=["alpha", "beta", "gamma", "lambda"], stim_color="royalblue", axes=None):
    """ """
    if axes is None:
        f, axes = plt.subplots(1, len(p), figsize=(1.5 * len(p), 3.5))
    for ax, p in zip(axes, p):
        cp.plot_group_by_stim(
            fit_df,
            y=p,
            ax=ax,
            stim_color=stim_color,
            print_stats=True,
            allow_neg=False,
            legend=False,
        )


def get_curve_fit_df(psy_curve_df, fit="sigmoid", **fit_kwargs):
    fit_results = []
    for subject in SUBJECT_IDS:
        for stim_trial in [True, False]:
            sub_df = psy_curve_df[(psy_curve_df.subject_ID == subject) & (psy_curve_df.stim_trial == stim_trial)]
            x = sub_df.habit_value.values.astype(float)
            y = sub_df.p_correct.values.astype(float)
            if fit == "sigmoid":
                fit_params = get_sigmoid_curve(x, y, return_as="params", **fit_kwargs)
            elif fit == "log_sigmoid":
                fit_params = get_log_sigmoid_curve(x, y, return_as="params", **fit_kwargs)
            else:
                raise ValueError("fit_model must be 'sigmoid' or 'log_sigmoid'")
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
        f, ax = plt.subplots(1, 1, figsize=(1, 1))
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
    for stim_trial, color in zip([False, True], ["grey", stim_color]):
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
            markersize=6,
            alpha=1,
        )
        ax.set_title(group)
    ax.set_xlim(*habit_value_range)

    return


def plot_habit_psychometrics_summary(
    psy_curve_df,
    stim_color="#0077FF",
    print_stats=True,
    fit="sigmoid",
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
        for stim_trial, color in zip([False, True], ["grey", stim_color]):
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
            if fit == "sigmoid":
                x_fit, y_fit = get_sigmoid_curve(x, y)
            elif fit == "log_sigmoid":
                x_fit, y_fit = get_log_sigmoid_curve(x, y)
            else:
                raise ValueError("fit_model must be 'sigmoid' or 'log_sigmoid'")
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
    decision_points_only=True,
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


def get_log_sigmoid_curve(
    x,
    y,
    return_as="line",
    eps=1e-3,
    x_fit="auto",
):
    """Fit a 4-parameter psychometric curve on a log-x axis.

    Model:
        y = gamma + (1 - gamma - lambda) * sigmoid(beta * (log(x+eps) - log(alpha+eps)))

    Parameters:
        alpha : threshold / x-shift in x-space (comparable across curves)
        beta  : steepness (slope in log-x space)
        gamma : lower asymptote (guess rate)
        lambda: lapse rate (upper asymptote = 1 - lambda)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 4:
        raise ValueError("Need at least 4 finite points to fit 4-parameter curve")

    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")

    x = np.clip(x, 0.0, 1.0)

    gamma0 = float(np.clip(np.nanmin(y), 0.0, 0.4))
    lam0 = float(np.clip(1.0 - np.nanmax(y), 0.0, 0.4))
    y_mid = gamma0 + 0.5 * (1.0 - gamma0 - lam0)
    try:
        alpha0 = float(x[np.nanargmin(np.abs(y - y_mid))])
    except ValueError:
        alpha0 = 0.5
    alpha0 = float(np.clip(alpha0, 0.05, 0.95))
    beta0 = 5.0

    p0 = [alpha0, beta0, gamma0, lam0]
    bounds = ([0.0, 0.0, 0.0, 0.0], [1.0, np.inf, 1.0, 0.49])

    popt, _pcov = curve_fit(
        lambda x_in, alpha, beta, gamma, lam: log_sigmoid_4p(x_in, alpha, beta, gamma, lam, eps=eps),
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100_000,
    )

    if x_fit == "auto":
        x_fit_vals = np.linspace(float(np.min(x)), float(np.max(x)), 500)
    else:
        x_fit_vals = np.linspace(0, 1, 500)
    y_fit = log_sigmoid_4p(x_fit_vals, *popt, eps=eps)

    if return_as == "line":
        return x_fit_vals, y_fit
    elif return_as == "params":
        return {"alpha": popt[0], "beta": popt[1], "gamma": popt[2], "lambda": popt[3]}
    else:
        raise ValueError("return_as must be 'line' or 'params'")


def log_sigmoid_4p(x, alpha, beta, gamma, lam, eps=1e-3):
    """4-parameter logistic in log-x space (log-logistic psychometric)."""
    x = np.asarray(x, dtype=float)
    eps = float(eps)
    x_log = np.log(np.clip(x, 0.0, None) + eps)
    alpha_log = np.log(np.clip(alpha, 0.0, None) + eps)
    z = beta * (x_log - alpha_log)
    z = np.clip(z, -500, 500)
    s = 1.0 / (1.0 + np.exp(-z))
    return gamma + (1.0 - gamma - lam) * s
