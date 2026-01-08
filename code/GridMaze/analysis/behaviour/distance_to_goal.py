"""
Analyses looking at rate of change of distance to goal aligned to cue, with and without stim across groups
@peterdoohan
"""

# %% Imports
import json
from tkinter import font
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import plotting as cp


# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

FRAME_RATE = 60

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as f:
    SUBJECT_IDS = json.load(f)

# %% curve fitting


def plot_curve_fit_random_effects_summary(curve_fit_results, print_stats=True, axes=None):
    """ """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(7, 3.5))
    for (
        p,
        color,
        ax,
    ) in zip(
        ["amplitude", "lambda", "offset"],
        ["#EC2E0C", "#EC7914", "#BEC50C"],
        axes,
    ):
        cp.plot_group_by_stim(
            curve_fit_results,
            y=p,
            ax=ax,
            stim_color=color,
            allow_neg=True if p == "offset" else False,
            print_stats=print_stats,
        )
        ax.set_title(p)


def get_ddtg_curve_fit_df(
    df,
    stim_day_range=(8, np.inf),
    ignore_sessions_with_issues_noted=False,
    t_range=(0, 4.0),
    plot=False,
):
    """
    Fits the rate of change of distance to goal curve for each subject (one curve for stim trials and one
    curve for non-stim trials), with an exponential decay function and returnes fitted parameters:
    delta_distance_to_goal(t) = A * exp(-lambda * t) + C

    where:
        A = amplitude
        lambda = decay rate
        C = offset
    """
    # filter data
    _df = _filter_ddtg_df(df, stim_day_range, ignore_sessions_with_issues_noted)

    # get average delta dtg to goal traces per subject
    subj_avg = (
        _df.groupby(
            [
                "subject_ID",
                "condition",
                "stim_trial",
            ]
        )
        .delta_distance_to_goal.mean()
        .delta_distance_to_goal
    )

    # loop over subjects
    results = []
    for subject in SUBJECT_IDS:
        subj_df = subj_avg.loc[subject]
        condition = subj_df.index.get_level_values(0).unique()[0]
        subj_df = subj_df.droplevel(0, axis=0)
        for stim_trial in [True, False]:
            trace = subj_df.loc[stim_trial]
            times = trace.index.astype(float)
            mask = (t_range[0] <= times) & (times < t_range[1])
            x = times[mask]
            y = trace.values[mask]
            # fit
            _results = {"subject_ID": subject, "condition": condition, "stim_trial": stim_trial}
            try:
                A, lambd, C = ddtg_curve_fit(
                    x,
                    y,
                    plot=plot,
                    subject_ID=subject,
                    stim_trial=stim_trial,
                )
                _results["amplitude"] = A
                _results["lambda"] = lambd
                _results["offset"] = C
            except RuntimeError:
                print(f"Fit failed for {subject} | {condition} | stim_trial={stim_trial}")
                _results["amplitude"] = np.nan
                _results["lambda"] = np.nan
                _results["offset"] = np.nan
            results.append(_results)
    return pd.DataFrame(results)


def ddtg_curve_fit(x, y, plot=False, subject_ID=None, stim_trial=None):
    """ """
    # -- initial guesses and bounds --
    p0 = [y.max() - y.min(), 1.0 / (x.max() - x.min()), y.min()]  # [A, lambda, C]
    lower_bounds = [0.0, 0.0, -np.inf]  # A>=0, lambda>=0
    upper_bounds = [np.inf, np.inf, np.inf]

    # -- Fit with curve_fit; provide sigma if available for weighting --
    (A, _lambda, C), pcov = curve_fit(
        exp_decay,
        x,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
    )
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.set_xlabel("Cue (s)")
        ax.set_ylabel("Δ DTG \n (cm/s)")
        ax.plot(x, y, label="data", color="k")
        ax.plot(x, exp_decay(x, A, _lambda, C), label="fit", color="r", ls="--")
        if subject_ID is not None and stim_trial is not None:
            ax.set_title(f"{subject_ID} | stim_trial={stim_trial}")
    return A, _lambda, C


def exp_decay(x, A, lambd, C):
    return A * np.exp(-lambd * x) + C


# %% plotting


def plot_delta_distance_to_goal_summary(
    df,
    stim_day_range=(8, np.inf),
    ignore_sessions_with_issues_noted=False,
    smooth_SD=0.2,  # alreadty smoothed
    t_range=(-5, 5),
    stim_color="#0077FF",
    axes=None,
):
    """ """
    # filter data
    _df = _filter_ddtg_df(df, stim_day_range, ignore_sessions_with_issues_noted)

    # get average delta distance to goal traces per subject
    subj_avg = (
        _df.groupby(
            [
                "condition",
                "stim_trial",
                "subject_ID",
            ]
        )
        .delta_distance_to_goal.mean()
        .delta_distance_to_goal
    )
    cond_grouped = subj_avg.groupby(level=[0, 1])
    mean = cond_grouped.mean()
    sem = cond_grouped.sem()

    # set up fig
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)
    for cond, ax in zip(["control", "opto"], axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Cue (s)")
        ax.set_title(cond)
    axes[0].set_ylabel("Δ distance-to-goal \n (cm/s)")

    for cond, ax in zip(["control", "opto"], axes):
        mean_cond = mean.loc[cond]
        sem_cond = sem.loc[cond]
        times = mean_cond.columns.astype(float)
        for stim_trial, color in zip(
            [True, False],
            [stim_color, "dimgrey"],
        ):
            _mean = mean_cond.loc[stim_trial].values
            _sem = sem_cond.loc[stim_trial].values
            if smooth_SD:
                _mean = gaussian_filter1d(_mean, sigma=smooth_SD * FRAME_RATE)
                _sem = gaussian_filter1d(_sem, sigma=smooth_SD * FRAME_RATE)
            ax.plot(times, _mean, label=stim_trial, color=color)
            ax.fill_between(times, _mean - _sem, _mean + _sem, alpha=0.2, color=color)
    for ax in axes:
        ax.set_xlim(t_range)
    axes[0].legend(title="light on", loc="upper right", fontsize="x-small")


def _filter_ddtg_df(
    df,
    stim_day_range=(8, np.inf),
    ignore_sessions_with_issues_noted=False,
):
    # filter data
    _df = df.copy()
    if stim_day_range is not None:
        _df = _df[_df.total_stim_days.between(*stim_day_range)]
    if ignore_sessions_with_issues_noted:
        _df = _df[~_df.session_issue_noted]
    return _df


# %%
def get_delta_distance_to_goal_df(
    sessions=None,
    verbose=False,
    jobs=-1,
    save=False,
):
    """ """
    # load and return if already generated
    save_path = RESULTS_PATH / "behaviour" / "delta_distance_to_goal_df.parquet"
    if not save and save_path.exists():
        if verbose:
            print("Loading delta_distance_to_goal_df from results...")
        df = pd.read_parquet(save_path)
        # fit multi-type multi-index columns
        info_df = df.drop(["delta_distance_to_goal"], axis=1, level=0)
        ddtg = df.xs("delta_distance_to_goal", axis=1, level=0)
        ddtg.columns = pd.MultiIndex.from_product([["delta_distance_to_goal"], ddtg.columns.astype(float)])
        return pd.concat([info_df, ddtg], axis=1)

    # else generate
    if sessions is None:
        # load all stim sessions
        if verbose:
            print("Loading all stim sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            stim_only=True,
            with_data=["trials_df", "navigation_df"],
            must_have_data=True,
            verbose=True,
        )
    # calc excess steps & other metrics for each session
    if jobs:
        dfs = Parallel(n_jobs=jobs)(delayed(get_session_delta_distance_to_goal_df)(session) for session in sessions)
    else:
        dfs = []
        for session in sessions:
            if verbose:
                print(session.name)
            _df = get_session_delta_distance_to_goal_df(session)
            dfs.append(_df)
    ddtg_df = pd.concat(dfs, ignore_index=True)

    # save to disk
    if save:
        if verbose:
            print("Saving delta_distance_to_goal_df to results...")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ddtg_df.to_parquet(save_path)
    return ddtg_df


def get_session_delta_distance_to_goal_df(
    session,
    cue_window=10,
    smooth_SD=0.1,
):
    """ """
    window_frames = cue_window * FRAME_RATE

    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    navigation_df = session.navigation_df
    skeleton_maze = session.skeleton_maze()
    label2coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    dD_dts = []
    for trial in trials_df.index.to_list():
        goal_coord = label2coord[trials_df.loc[trial, ("goal", "")] + "_C"]
        trial_df = navigation_df[navigation_df.trial == trial]
        if trial_df.empty:
            dD_dts.append(np.full((window_frames * 2,), np.nan))
            continue
        cue_indx = navigation_df[navigation_df.trial == trial].index[0]
        locs = navigation_df.iloc[
            cue_indx - window_frames : cue_indx + window_frames + 1
        ].maze_position.skeleton.to_numpy()
        if len(locs) != (window_frames * 2 + 1):
            dD_dts.append(np.full((window_frames * 2,), np.nan))
            continue  # window out of session bounds
        coords = [label2coord[loc] for loc in locs]
        aligned_dtg = np.array([shortest_path_lengths[c][goal_coord] for c in coords])
        dD_dt = np.diff(aligned_dtg) * FRAME_RATE  # m/s
        dD_dts.append(dD_dt)
    D = np.vstack(dD_dts)  # [trials, timepoints]
    if smooth_SD:
        D = gaussian_filter1d(D, sigma=smooth_SD * FRAME_RATE, axis=1)
    # warp into dataframe for convenience
    aligned_times = np.arange(-cue_window, cue_window + 1 / FRAME_RATE, 1 / FRAME_RATE)[:-1]
    ddtg_df = pd.DataFrame(
        D,
        columns=pd.MultiIndex.from_product([["delta_distance_to_goal"], aligned_times.tolist()]),
    )
    info_df = trials_df.reset_index()[[("trial", ""), ("stim_trial", "")]]
    info_df[("subject_ID", "")] = session.subject_ID
    info_df[("condition", "")] = session.condition
    info_df[("maze_name", "")] = session.maze_name
    info_df[("day_on_maze", "")] = session.day_on_maze
    info_df[("stim_day", "")] = session.stim_day
    info_df[("total_stim_days", "")] = session.total_stim_days
    info_df[("session_issue_noted", "")] = True if session.session_notes is not None else False

    return pd.concat([info_df, ddtg_df], axis=1)
