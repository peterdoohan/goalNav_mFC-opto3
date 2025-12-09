"""
Behaviouoral analysis aligned to stim onset.
@peterdoohan
"""

# %% Imports
import pandas as pd
import networkx as nx
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, zscore
from ..core import get_sessions as gs
from scipy.ndimage import gaussian_filter1d

# %% Global Variables
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

SUBJECT_INFO = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

FRAME_RATE = 60  # Hz

# %% Speed


def get_stim_aligned_speed(group="opto", window=(-1, 2)):
    """"""
    valid_subjects = SUBJECT_INFO[
        (SUBJECT_INFO.condition == group) & (SUBJECT_INFO.included_in_full_trial_stim)
    ].subject_ID
    stim_aligned_speeds = []
    for subject in valid_subjects:
        sessions = gs.get_sessions(
            experiment_phases="full_trial_stim", subject_IDs=[subject], with_data=["navigation_df", "trials_df"]
        )
        speeds = _get_stim_aligned_speed(sessions, window=window)
        stim_aligned_speeds.append(np.mean(speeds, axis=0))
    # normalise speeds
    stim_aligned_speeds = zscore(stim_aligned_speeds, axis=1)
    av_speed = np.mean(stim_aligned_speeds, axis=0)
    sem_speeds = sem(stim_aligned_speeds, axis=0)
    # plot results
    f, ax = plt.subplots(figsize=(4, 4), clear=True)
    aligned_time = np.linspace(window[0], window[1], len(av_speed))
    ax.plot(aligned_time, av_speed)
    ax.fill_between(aligned_time, av_speed - sem_speeds, av_speed + sem_speeds, alpha=0.5)
    for i in range(len(valid_subjects)):
        ax.plot(aligned_time, stim_aligned_speeds[i], alpha=0.1, color="black")
    ax.axvline(0, color="black", linestyle="--")
    return stim_aligned_speeds


def _get_stim_aligned_speed(sessions, window=(0, 2)):
    """
    Get the speed of the animal aligned to the onset of opto stim events.
    Args:
        session (MazeSession): session object, must have with_data=["navigation_df"]
        window (tuple): The time window to extract around the stim onset in seconds.
    """
    window = np.multiply(window, FRAME_RATE)
    speeds = []
    for session in sessions:
        navigation_df = session.navigation_df
        speed = navigation_df.speed
        trials = _get_trials(session, stim=True)
        for t in trials:
            trial_df = navigation_df[navigation_df.trial == t]
            stim_onset_frame = trial_df[trial_df.stim_on].index[0]
            start_frame = stim_onset_frame + window[0] + 1
            end_frame = stim_onset_frame + window[1]
            stim_aligned_speed = speed.loc[start_frame:end_frame].to_numpy()
            speeds.append(stim_aligned_speed)
    return np.vstack(speeds)


# %% Rate of change of distance to gola (delta-dtg)


def _get_trials(session, stim):
    """Returns list of trials in the stim_off or stim_on conditions."""
    trials_df = session.trials_df
    x = "full_trial" if stim else False
    return trials_df[trials_df.stim_trial == x].trial.to_list()


def get_session_rate_of_change_of_distance_to_goal(session, cue_window=3, stim=False):
    """"""
    navigation_df = session.navigation_df
    skeleton_maze = session.skeleton_maze()
    window_frames = cue_window * FRAME_RATE
    skeleton_label2skeleton_coord = {v: k for k, v in nx.get_node_attributes(skeleton_maze, "label").items()}
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(skeleton_maze, weight="weight"))
    trial_dD_dts = []
    trials = _get_trials(session, stim)
    for trial in trials:
        trial_df = navigation_df[navigation_df.trial == trial]
        goal = trial_df.goal.unique()[0]
        goal_coord = skeleton_label2skeleton_coord[goal + "_C"]
        cue_indx = trial_df.index[0]
        sk_locations = navigation_df.iloc[
            cue_indx - window_frames : cue_indx + window_frames + 1
        ].maze_position.skeleton.to_numpy()
        if len(sk_locations) == 0:
            continue  # window out of session bounds
        sk_coords = [skeleton_label2skeleton_coord[loc] for loc in sk_locations]
        geo_distance_to_goal = np.array([shortest_path_lengths[c][goal_coord] for c in sk_coords])
        dD_dt = np.diff(geo_distance_to_goal) * FRAME_RATE  # m/s
        trial_dD_dts.append(dD_dt)
    return np.vstack(trial_dD_dts)  # [trials, timepoints]


def get_subject_rate_of_change_of_distance_to_goal(subject, window_length, stim):
    sessions = gs.get_sessions(
        experiment_phases="full_trial_stim",
        subject_IDs=[subject],
        with_data=["navigation_df", "trials_df"],
    )
    subject_dD_dts = []
    for session in sessions:
        session_dD_dts = get_session_rate_of_change_of_distance_to_goal(session, cue_window=window_length, stim=stim)
        subject_dD_dts.append(session_dD_dts)
    return np.vstack(subject_dD_dts)  # [sessions * trials, timepoints]


def plot_cross_subject_rate_of_change_of_distance_to_goal(
    group="opto", window_length=6, smooth_SD=8, plot_individual=False, ax=None
):
    subjects = SUBJECT_INFO[(SUBJECT_INFO.condition == group) & (SUBJECT_INFO.included_in_full_trial_stim)].subject_ID
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(3, 3), clear=True)
    for stim, color in zip([False, True], ["grey", "deepskyblue"]):
        dD_dts = []
        for subject in subjects:
            subject_dD_dts = get_subject_rate_of_change_of_distance_to_goal(
                subject, window_length=window_length, stim=stim
            )
            av_subject_dD_dt = np.mean(gaussian_filter1d(subject_dD_dts, sigma=smooth_SD), axis=0)
            dD_dts.append(np.array(av_subject_dD_dt))
        av_dD_dt = np.mean(dD_dts, axis=0)
        sem_dD_dt = np.std(dD_dts, axis=0) / np.sqrt(len(subjects))
        time = np.linspace(-window_length, window_length, len(av_dD_dt))
        # plotting
        if plot_individual:
            for d in dD_dts:
                ax.plot(time, d, color=color, lw=0.5, alpha=0.5)
        ax.plot(time, av_dD_dt, color=color, lw=2, label=f"Stim: {stim}")
        ax.fill_between(
            time,
            av_dD_dt - sem_dD_dt,
            av_dD_dt + sem_dD_dt,
            color=color,
            alpha=0.3,
        )
    ax.axvline(0, color="silver", linestyle="--")
    ax.set_xlabel("Cue-aligned Time (s)")
    ax.set_ylabel("Δ Shortest Path Distance \n to Goal (m/s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    f.tight_layout()
