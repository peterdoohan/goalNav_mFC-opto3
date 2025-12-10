"""
Lib for quantification of movement dynamics between groups (opto & control)
@peterdoohan
"""

# %% Imports
import json
from turtle import pos
from itsdangerous import NoneAlgorithm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from regex import F
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables
from GridMaze.paths import ANALYSIS_INFO_PATH

with open(ANALYSIS_INFO_PATH / "subject_movement_thresholds.json", "r") as infile:
    SUBJECT_MOVEMENT_THRESHOLDS = json.load(infile)


FRAME_RATE = 60

# %% Functions


def get_stationary_bouts_df(sessions=None, verbose=True):
    """ """
    if sessions is None:
        if verbose:
            print("Loading sessions...")
        sessions = gs.get_maze_sessions(
            subject_IDs="all",
            conditions="all",
            stim_only=True,
            with_data=["navigation_df"],
            must_have_data=True,
            verbose=verbose,
        )
    bouts_dfs, stationary_dfs = [], []
    for session in sessions:
        if verbose:
            print(session.name)
        _stationary_df, _bouts_df = get_session_stationary_bouts_dfs(session)
        stationary_dfs.append(_stationary_df)
        bouts_dfs.append(_bouts_df)

    # combine data
    stationary_df = pd.concat(stationary_dfs, ignore_index=True)
    bouts_df = pd.concat(bouts_dfs, ignore_index=True)
    return stationary_df, bouts_df


def get_session_stationary_bouts_dfs(session, speed_smooth_SD=5, min_bout_duration=1.0, clip_trial_times=(0.5, 0.5)):
    """
    Quantify the number of long stationary bouts during light on periods (stim)
    for a given session.
    """
    # load data
    navigation_df = session.navigation_df

    trials = navigation_df.trial.dropna().unique()
    bout_results = []
    trial_results = []
    for trial in trials:

        trial_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
        # clip start and end which can be stationary due to reward consumption
        if clip_trial_times:
            try:
                trial_df = trial_df.iloc[int(clip_trial_times[0] * FRAME_RATE) : -int(clip_trial_times[1] * FRAME_RATE)]
            except IndexError:
                continue  # short trial not informative

        if len(trial_df) == 0:
            continue

        trial_unique_ID = trial_df.trial_unique_ID.unique()[0]
        stim_trial = trial_df.stim_on.any()

        # comparing moving vs stationary
        speed = trial_df.speed.values
        if speed_smooth_SD:
            speed = gaussian_filter1d(speed, sigma=speed_smooth_SD)
        moving = speed >= SUBJECT_MOVEMENT_THRESHOLDS[session.subject_ID]

        trial_duration = len(trial_df) / FRAME_RATE
        total_stationary_time = np.sum(~moving) / FRAME_RATE
        stationary_frac = total_stationary_time / trial_duration
        trial_results.append(
            {
                "trial_unique_ID": trial_unique_ID,
                "stim_trial": stim_trial,
                "trial_duration": trial_duration,
                "total_stationary_time": total_stationary_time,
                "stationary_frac": stationary_frac,
            }
        )

        # quantify moving bouts
        padded = np.r_[False, moving, False]
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]  # indices in padded -> correspond to original indices
        ends = np.where(diff == -1)[0]
        durations = (ends - starts) / FRAME_RATE  # in seconds

        masks = []  # to define each bout in the trial
        for s, e, d in zip(starts, ends, durations):
            if d >= int(min_bout_duration):
                mask = np.zeros_like(moving, dtype=bool)
                mask[s:e] = True
                masks.append(mask)

        for i, mask in enumerate(masks):
            bout_df = trial_df.loc[mask]
            bout_results.append(
                {
                    "trial_unique_ID": trial_unique_ID,
                    "stim_trial": stim_trial,
                    "bout_number": i,
                    "bout_duration": len(bout_df) / FRAME_RATE,
                    "stim_on": bout_df.stim_on.any(),
                }
            )

    stationary_df = pd.DataFrame(trial_results)
    bouts_df = pd.DataFrame(bout_results)

    # add session info
    for info in ["subject_ID", "condition", "maze_name", "maze_order", "day_on_maze", "stim_day", "total_stim_days"]:
        stationary_df[info] = getattr(session, info)
        bouts_df[info] = getattr(session, info)

    return stationary_df, bouts_df


def plot_trial_speed_profiles(session, trial, speed_smooth_SD=4, ax=None):
    """ """
    # set up figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("time in trial (s)")
    ax.set_ylabel("speed (m/s)")

    # load data
    navigation_df = session.navigation_df

    # filter for trial
    trial_df = navigation_df[(navigation_df.trial == trial) & (navigation_df.trial_phase == "navigation")]
    start_time = trial_df.time.values[0]
    time = trial_df.time.values - start_time
    speed = trial_df.speed.values
    if speed_smooth_SD:
        speed = gaussian_filter1d(speed, sigma=speed_smooth_SD)

    # get times leaving start loc and entering goal loc
    locs = trial_df.maze_position.simple
    moved_loc_frames = locs[locs != locs.shift(1)].index
    moved_start = trial_df.loc[moved_loc_frames[1], ("time", "")] - start_time
    moved_goal = trial_df.loc[moved_loc_frames[-1], ("time", "")] - start_time

    # plot
    stim_mask = trial_df.stim_on.values
    if stim_mask.any():
        ax.plot(time[stim_mask], speed[stim_mask], color="#0077FF", lw=5, alpha=0.4)
    ax.plot(time, speed, color="black", lw=1)
    ax.axhline(3 * SUBJECT_MOVEMENT_THRESHOLDS[session.subject_ID], color="red", ls="--", lw=1, alpha=0.5)
    ax.axvline(moved_start, color="green", ls="--", lw=1, alpha=0.5)
    ax.axvline(moved_goal, color="gold", ls="--", lw=1, alpha=0.5)
