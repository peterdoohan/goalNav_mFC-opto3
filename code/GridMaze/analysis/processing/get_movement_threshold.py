"""This modules analyses the distribution of speeds through the experiment to derive a threshold for when the animals are moving (vs stationary)"""

# %% Imports
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from GridMaze.analysis.core import get_sessions as gs
from GridMaze.analysis.core import load_data
from sklearn.mixture import GaussianMixture

# %% Global variables
from GridMaze.paths import PROCESSED_DATA_PATH, ANALYSIS_INFO_PATH

SMALL_CONSTANT = 0.0001
# %% Main function


def save_movement_threshold():
    movement_thresholds = get_movement_threshold(plot=False, return_grand_average=False)
    with open(ANALYSIS_INFO_PATH / "subject_movement_thresholds.json", "w") as outfile:
        json.dump(movement_thresholds, outfile)
    movement_threshold = np.mean(list(movement_thresholds.values()))
    with open(ANALYSIS_INFO_PATH / "movement_threshold.json", "w") as outfile:
        json.dump(movement_threshold, outfile)

    return print(f"Movement threshold saved to analysis_info")


def get_movement_threshold(plot=True, smooth_SD=0.06, return_grand_average=False):
    subject_IDs = [f.name for f in PROCESSED_DATA_PATH.iterdir() if f.is_dir()]
    subject_movement_thresholds = []
    for subject in subject_IDs:
        subject_sessions = [f for f in (PROCESSED_DATA_PATH / subject).iterdir() if f.is_dir()]
        subject_speeds = []
        for session in subject_sessions:
            trajectories_df = load_data.load(session / "frames.trajectories.htsv")
            if trajectories_df is None:
                continue
            velocities = get_velocities(trajectories_df, smooth_SD=smooth_SD)
            speeds = get_speeds(velocities)
            subject_speeds.append(speeds)
        if subject_speeds == []:
            continue
        subject_speeds = np.hstack(subject_speeds)
        log_speeds = np.log(speeds + SMALL_CONSTANT)
        log_speeds = log_speeds.reshape(-1, 1)
        GM = GaussianMixture(n_components=2).fit(log_speeds)
        x = np.linspace(log_speeds.min(), log_speeds.max(), 1000).reshape(-1, 1)
        assignments = GM.predict(x).astype(bool)
        threshold = max(x[np.where(assignments == 0)][0][0], x[np.where(assignments == 1)][0][0])
        subject_movement_thresholds.append(np.exp(threshold))
        if plot:
            logprob = GM.score_samples(x)
            pdf = np.exp(logprob)
            f, ax = plt.subplots(1, 2, clear=True)
            ax[0].set_title(subject)
            ax[0].hist(log_speeds, bins=1000, density=True, alpha=0.5)
            ax[0].plot(x, pdf, "-r")
            ax[0].axvline(threshold, color="green")
            ax[0].set_xlabel("log speed")
            ax[0].set_xlim(-10, 0)
            ax[1].hist(speeds, bins=1000, density=True, alpha=0.5)
            ax[1].axvline(np.exp(threshold), color="green")
            ax[1].set_xlabel("speed")
            ax[1].set_xlim(0, 0.6)
    if return_grand_average:
        return np.mean(subject_movement_thresholds)
    else:
        return {subj: float(thres) for subj, thres in zip(subject_IDs, subject_movement_thresholds)}


def get_velocities(trajectories_df, smooth_SD=0.03, frame_rate=60):
    """Calculates frame velocities from trajectories_df by first smoothing centroid coordinates to remove high frequency noise
    from DLC position estimation:
    - INPUTS:
        - trajectories_df: loaded from processed data
        - smooth_SD: standard deviation of gaussian filter used to smooth centroid coordinates (in seconds)
        - frame_rate: frame rate of video aquisition, time between rows/frames (in Hz)
    """
    smooth_SD = smooth_SD * frame_rate  # convert to frames
    # extract raw data from trajectories_df
    x = trajectories_df.centroid_position.x.to_numpy()
    y = trajectories_df.centroid_position.y.to_numpy()
    times = trajectories_df.time.to_numpy()
    # apply smoothing to remove high frequency noise from DLC position estimation
    smooth_x = gaussian_filter1d(x, sigma=smooth_SD)
    smooth_y = gaussian_filter1d(y, sigma=smooth_SD)
    # calculate velocities from smoothed data
    dt = np.diff(times)
    dx = np.diff(smooth_x)
    dy = np.diff(smooth_y)
    vx = dx / dt
    vy = dy / dt
    velocity = np.column_stack((vx, vy))
    # Linear extrapolation for the last velocity value
    last_vx = vx[-1] + (vx[-1] - vx[-2])
    last_vy = vy[-1] + (vy[-1] - vy[-2])
    last_velocity = np.array([last_vx, last_vy]).reshape(1, 2)
    velocity = np.append(velocity, last_velocity, axis=0)
    return velocity


def get_speeds(velocities):
    """Calculates speed from velocities"""
    return np.linalg.norm(velocities, axis=1)
