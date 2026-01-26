"""
Lib for quantification of movement dynamics between groups (opto & control)
@peterdoohan
"""

# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import RESULTS_PATH

FRAME_RATE = 60

MAX_STIM_DURATION = 30  # seconds


# %%
def plot_distance_travelled_summary(distance_traveled_df, stim_color="#0077FF", print_stats=True, axes=None):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(5, 2))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    # plot subject averages across conditions
    df = distance_traveled_df.copy()
    ys = ["total_distance", "average_speed", "stim_duration"]
    subj_avg = df.groupby(["subject_ID", "condition"])[ys].mean().reset_index()

    for ax, y in zip(axes, ys):
        sns.stripplot(
            data=subj_avg,
            y=y,
            hue="condition",
            palette=[stim_color, "black"],
            alpha=0.25,
            dodge=0.01,
            ax=ax,
            legend=False,
        )
        # plot subject mean & sem
        sns.pointplot(
            data=subj_avg,
            y=y,
            hue="condition",
            palette=[stim_color, "black"],
            errorbar="se",
            dodge=0.4,
            ax=ax,
            legend=False,
        )
        ax.set_ylim(bottom=0)

    if print_stats:
        for y in ys:
            opto_data = subj_avg[subj_avg.condition == "opto"][y]
            control_data = subj_avg[subj_avg.condition == "control"][y]
            t_stat, p_val = ttest_ind(opto_data, control_data)
            print(f"{y}: t={t_stat:.2f}, p={p_val:.4f}")


def get_distance_traveled_df(
    smooth_SD=0.1,
    sessions=None,
    jobs=-1,
    save=False,
    verbose=False,
):
    """ """
    # load and return if already generated
    save_path = RESULTS_PATH / "behaviour" / f"distance_traveled.parquet"
    if not save and save_path.exists():
        if verbose:
            print("Loading df from results...")
        performance_df = pd.read_parquet(save_path)
        return performance_df

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
        dfs = Parallel(n_jobs=jobs)(
            delayed(get_session_distance_traveled_df)(
                session,
                smooth_SD=smooth_SD,
                verbose=verbose,
            )
            for session in sessions
        )
    else:
        dfs = []
        for session in sessions:
            if verbose:
                print(session.name)
            _df = get_session_distance_traveled_df(
                session,
                smooth_SD=smooth_SD,
                verbose=verbose,
            )
            dfs.append(_df)
    distance_traveled_df = pd.concat(dfs, ignore_index=True)

    # save to disk
    if save:
        if verbose:
            print("Saving distance_traveled_df to results...")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        distance_traveled_df.to_parquet(save_path)
    return distance_traveled_df


def get_session_distance_traveled_df(session, smooth_SD=0.1, verbose=False):
    """ """
    # load data
    navigation_df = session.navigation_df.copy()
    times = navigation_df.time
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    trial2unique_ID = navigation_df.set_index("trial").trial_unique_ID.dropna().drop_duplicates().to_dict()

    # filter for stim trials
    stim_trials_df = trials_df[trials_df.stim_trial]

    results = []
    for trial in stim_trials_df.index:
        if trial not in trial2unique_ID.keys():
            if verbose:
                print(f"session {session.name} trial {trial} missing from navigation df")
            continue
        stim_on_time = stim_trials_df.loc[trial, ("time", "stim_start")]
        stim_off_time = stim_trials_df.loc[trial, ("time", "stim_end")]
        stim_duration = stim_off_time - stim_on_time
        stim_on_frame = (times - stim_on_time).abs().argmin()
        stim_off_frame = (times - stim_off_time).abs().argmin()
        if not stim_on_frame in navigation_df.index or not stim_off_frame in navigation_df.index:
            continue  # skip if frames not in navigation df
        trial_df = navigation_df.loc[stim_on_frame:stim_off_frame, :]
        X = trial_df.centroid_position.x.values
        Y = trial_df.centroid_position.y.values
        if smooth_SD:
            X = gaussian_filter1d(X, sigma=smooth_SD * FRAME_RATE)
            Y = gaussian_filter1d(Y, sigma=smooth_SD * FRAME_RATE)
        distances = np.sqrt(np.diff(X) ** 2 + np.diff(Y) ** 2)
        total_distance = np.sum(distances)
        av_speed = total_distance / stim_duration
        results.append(
            {
                "trial_unique_ID": trial2unique_ID[trial],
                "stim_duration": stim_duration,
                "total_distance": total_distance,
                "average_speed": av_speed,
            }
        )
    df = pd.DataFrame(results)
    df["subject_ID"] = session.subject_ID
    df["condition"] = session.condition
    df["maze_name"] = session.maze_name
    df["day_on_maze"] = session.day_on_maze
    df["total_stim_days"] = session.total_stim_days
    return df


# %% Stim-aligned speeds functions


def test(sessions):
    stim_on_df = get_aligned_speeds_df(aligned_to="stim_on", window=(-2, 6), smooth_SD=0.15, sessions=sessions)
    stim_off_df = get_aligned_speeds_df(aligned_to="stim_off", window=(-6, 2), smooth_SD=0.15, sessions=sessions)
    plot_stim_aligned_speeds_summary(stim_on_df, stim_off_df)
    return stim_on_df, stim_off_df


def plot_stim_aligned_speeds_summary(stim_on_df, stim_off_df, axes=None):
    """ """
    # set up figure
    if axes is None:
        f, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Speed (m/s)")
    for ax in axes:
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    # plot speed profiles
    for df, ax, label in zip([stim_on_df, stim_off_df], axes, ["stim on", "stim off"]):
        # get mean and sem across subjects in each condition
        subject_grouped = df.groupby(["subject_ID", "condition"]).speed.mean().speed.groupby(level=1)
        mean = subject_grouped.mean()
        sem = subject_grouped.sem()
        times = mean.columns.values.astype(float)
        for condition, color in zip(["control", "opto"], ["grey", "royalblue"]):
            _mean = mean.loc[condition].values.astype(float)
            _sem = sem.loc[condition].values.astype(float)
            ax.plot(times, _mean, label=condition, color=color)
            ax.fill_between(
                times,
                _mean - _sem,
                _mean + _sem,
                color=color,
                alpha=0.25,
            )
        ax.set_xlabel(label)
    axes[0].legend(fontsize="small")


def get_aligned_speeds_df(
    aligned_to="stim_on",
    window=(-2, 6),
    smooth_SD=0.1,
    sessions=None,
    jobs=-1,
    save=False,
    verbose=False,
):
    """ """
    # load and return if already generated
    save_path = RESULTS_PATH / "behaviour" / f"{aligned_to}_aligned_speeds.parquet"
    if not save and save_path.exists():
        if verbose:
            print("Loading df from results...")
        performance_df = pd.read_parquet(save_path)
        return performance_df

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
        dfs = Parallel(n_jobs=jobs)(
            delayed(get_session_aligned_speeds_df)(
                session,
                aligned_to=aligned_to,
                window=window,
                smooth_SD=smooth_SD,
            )
            for session in sessions
        )
    else:
        dfs = []
        for session in sessions:
            if verbose:
                print(session.name)
            _df = get_session_aligned_speeds_df(
                session,
                aligned_to=aligned_to,
                window=window,
                smooth_SD=smooth_SD,
            )
            dfs.append(_df)
    speeds_df = pd.concat(dfs, ignore_index=True)

    # save to disk
    if save:
        if verbose:
            print("Saving speeds_df to results...")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        speeds_df.to_parquet(save_path)
    return speeds_df


def get_session_aligned_speeds_df(
    session,
    aligned_to="stim_on",
    window=(-1, 8),
    smooth_SD=0.15,
):
    assert aligned_to in ["stim_on", "stim_off"], "aligned_to must be 'stim_on' or 'stim_off'"
    # load data
    navigation_df = session.navigation_df.copy()
    times = navigation_df.time
    trials_df = session.trials_df.copy()
    trials_df.set_index("trial", inplace=True)
    trial2unique_ID = navigation_df.set_index("trial").trial_unique_ID.dropna().drop_duplicates().to_dict()

    # filter for stim trials
    stim_trials_df = trials_df[trials_df.stim_trial]

    # define window in frames
    frames_before = int(abs(window[0]) * FRAME_RATE)
    frames_after = int(abs(window[1]) * FRAME_RATE)

    total_frames = frames_before + frames_after + 1
    S = np.zeros((len(stim_trials_df), total_frames))
    for i, trial in enumerate(stim_trials_df.index):
        # get nearest frame to align speed
        if aligned_to == "stim_on":
            align_time = stim_trials_df.loc[trial, ("time", "stim_start")]
        else:
            align_time = stim_trials_df.loc[trial, ("time", "stim_end")]
        align_frame = (times - align_time).abs().argmin()
        speeds = navigation_df.loc[align_frame - frames_before : align_frame + frames_after, "speed"].values
        if len(speeds) < total_frames:
            # pad with nans if at start or end of session
            speeds = np.pad(
                speeds,
                (0, total_frames - len(speeds)),
                mode="constant",
                constant_values=np.nan,
            )
        S[i, :] = speeds

    # smooth speed
    if smooth_SD:
        S = gaussian_filter1d(S, sigma=smooth_SD * FRAME_RATE, axis=1)

    # output as df
    aligned_times = np.linspace(window[0], window[1], frames_before + frames_after + 1)
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([["speed"], aligned_times]), data=S)
    df[("subject_ID", "")] = session.subject_ID
    df[("condition", "")] = session.condition
    df[("maze_name", "")] = session.maze_name
    df[("day_on_maze", "")] = session.day_on_maze
    df[("total_stim_days", "")] = session.total_stim_days
    df[("trial_unique_ID", "")] = stim_trials_df.index.map(trial2unique_ID)

    return df
