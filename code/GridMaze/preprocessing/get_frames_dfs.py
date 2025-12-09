""" """

# %% Imports
import numpy as np
import json
import h5py
import pandas as pd
from pathlib import Path
from datetime import datetime as dt
from GridMaze.preprocessing import pixels_to_position as pix2pos
from GridMaze.preprocessing import maze_registration as maze_reg
from GridMaze.preprocessing import rsync
from GridMaze.preprocessing import get_head_direction as hd
from GridMaze.preprocessing import get_maze_trajectories as mt

from GridMaze.preprocessing import get_pycontrol_dfs as pydf

from GridMaze.maze import representations as mr


# %% Global variables
from GridMaze.paths import SLEAP_PATH, EXPERIMENT_INFO_PATH


IMAGE_SIZE = maze_reg.get_image_size_from_video()
FRAME_RATE = 60  # Hz
with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as fp:
    SUBJECT_IDS = json.load(fp)

try:
    with open(EXPERIMENT_INFO_PATH / "bodypart_outlier_thresholds.json", "r") as fp:
        BODYPART_OUTLIER_THRESHOLDS = json.load(fp)
except FileNotFoundError:
    print("Bodypart outlier thresholds not found. Run save_bodypart_outlier_thresholds() to generate.")

try:
    with open(EXPERIMENT_INFO_PATH / "maze_measurements.json", "r") as fp:
        MAZE_MEASUREMENTS = json.load(fp)
except FileNotFoundError:
    print("Maze measurements not found. Check get_experiment_info().")


# %% Top level processing functions


def get_tracking_df(session_dir):
    """Generates the frames.tracking.htsv (dataframe) data strucutre containing tracking data for each mouse
    body part throughout a non-rest session. The tracking data is extracted from the SLEAP output file and
    positions are cleaned and translated to physical coordinates."""
    sleap_df = get_SLEAP_output_as_df(session_dir.SLEAP_path)
    sleap_df = translate_to_physical_coords(sleap_df, session_dir.big_maze_rig, session_dir.date)
    sleap_df = remove_jump_points(sleap_df)
    sleap_df = remove_far_away_points(sleap_df)
    return sleap_df


def get_trajectories_df(session_dir, centroid_part="head_back"):
    """
    Generates the frames.trajectories.htsv (dataframe) data structure containing the centroid position and head direction
    for open field and object open field sessions, as well as the maze position (simple, skeleton) for maze sessions.
    """
    assert session_dir.session_type != "rest", "frames.trajectories.htsv cannot be generated for rest sessions."
    base_trajectories_df = _get_base_trajectories_df(session_dir, centroid_part)
    if session_dir.session_type == "open_field":
        return base_trajectories_df
    elif session_dir.session_type == "maze":
        centroid = base_trajectories_df.centroid_position.drop(columns="interpolated")
        simple_maze = mr.get_simple_maze(session_dir.maze_name)
        skeleton_maze = mr.get_skeleton_maze(session_dir.maze_name)
        simple_maze_trajectory = mt.get_valid_simple_maze_trajectory(centroid, simple_maze)
        skeleton_maze_trajectory = mt.get_skeleton_maze_trajectory(centroid, skeleton_maze)
        maze_trajectories_df = pd.DataFrame(
            index=base_trajectories_df.index,
            data={
                ("maze_position", "simple"): simple_maze_trajectory,
                ("maze_position", "skeleton"): skeleton_maze_trajectory,
            },
        )
        columns = pd.MultiIndex.from_tuples(
            [(col, "") if isinstance(col, str) else col for col in maze_trajectories_df.columns],
        )
        maze_trajectories_df.columns = columns
        return pd.concat((base_trajectories_df, maze_trajectories_df), axis=1)


def get_trial_info_df(session_dir):
    tracking_df = get_tracking_df(session_dir)
    trials_df = pydf.get_trials_df(session_dir)
    frame_times = get_frame_pytimes(session_dir.video_sync_path, session_dir.pycontrol_path)
    if len(frame_times) > len(tracking_df):  # sometimes sleap output is missing from end of session
        frame_times = frame_times[: len(tracking_df)]
    elif len(tracking_df) > len(frame_times):  # catch cases where sync pulses stop early
        frame_times = _extend_times(frame_times, len(tracking_df))
    session_trials = convert_times2trials(trials_df, frame_times)
    trial2goal = {k: v for k, v in zip(trials_df.trial.to_numpy(), trials_df.goal.to_numpy())}
    session_goals = pd.Series(session_trials).map(trial2goal).to_numpy()
    trial_phase = convert_times2trial_phases(trials_df, frame_times)
    stim_on = convert_times2stim_on(trials_df, frame_times)
    trial_info_df = pd.DataFrame(
        {"trial": session_trials, "trial_phase": trial_phase, "goal": session_goals, "stim_on": stim_on}
    )
    return trial_info_df


# %%


def convert_times2trials(trials_df, session_times):
    """Converts session times to trial numbers active at that timepoint"""
    trial_array = np.full(len(session_times), np.nan, dtype=float)
    trial_starts = trials_df["time"]["cue"].values
    trial_ends = trials_df["time"]["trial_end"].values
    trial_numbers = trials_df["trial"].values
    trial_indices = np.searchsorted(trial_starts, session_times, side="right") - 1
    valid_indices = np.logical_and(trial_indices >= 0, session_times <= trial_ends[trial_indices])
    trial_array[valid_indices] = trial_numbers[trial_indices[valid_indices]]
    return trial_array


def convert_times2trial_phases(trials_df, session_times):
    cue_times = trials_df.time["cue"].values
    reward_times = trials_df.time["reward"].values
    end_reward_consumption_times = trials_df.time["end_reward_consumption"].values
    trial_end_times = trials_df.time["trial_end"].values
    navigation_mask = np.logical_and(cue_times <= session_times[:, None], session_times[:, None] < reward_times)
    reward_consumption_mask = np.logical_and(
        reward_times <= session_times[:, None], session_times[:, None] < end_reward_consumption_times
    )
    ITI_mask = np.logical_and(
        end_reward_consumption_times <= session_times[:, None], session_times[:, None] <= trial_end_times
    )
    phases = np.full(len(session_times), np.nan, dtype="object")
    phases[navigation_mask.any(axis=1)] = "navigation"
    phases[ITI_mask.any(axis=1)] = "ITI"
    phases[reward_consumption_mask.any(axis=1)] = "reward_consumption"
    return phases


def convert_times2stim_on(trials_df, frame_times):
    """ """
    stim_bool = np.full(len(frame_times), False)
    stim_trials_df = trials_df[trials_df.stim_trial != False]
    if len(stim_trials_df) == 0:
        return stim_bool
    else:
        stim_start_times = stim_trials_df.time.stim_start.values
        stim_end_times = stim_trials_df.time.stim_end.values
        stim_mask = np.logical_and(stim_start_times[:, None] <= frame_times, frame_times <= stim_end_times[:, None])
        stim_bool[stim_mask.any(axis=0)] = True
    return stim_bool


# %% supporting functions


def translate_to_physical_coords(dlc_df, big_maze_rig, session_date):
    """Translate pixel coordinate x,y pixel positions in dlc_df to physical coordinates (meters)."""
    body_parts = dlc_df.columns.get_level_values(0).unique()
    for part in body_parts:
        pix_coords = dlc_df[[(part, "x"), (part, "y")]].to_numpy()
        physical_coords = pix2pos.translate_pixel2physical_coords(pix_coords, big_maze_rig, session_date)
        dlc_df[[(part, "x"), (part, "y")]] = physical_coords
    return dlc_df


def _get_base_trajectories_df(session_directory, centroid_part):
    """ """
    tracking_df = get_tracking_df(session_directory)
    centroid_tracking = tracking_df[centroid_part]
    centroid_interpolated = centroid_tracking.isna().all(axis=1).astype(int)  # frames that will be interpolated
    centroid = centroid_tracking.interpolate(method="linear", limit_direction="both", limit=100000, axis=0)
    pycontrol_times = get_frame_pytimes(session_directory.video_sync_path, session_directory.pycontrol_path)
    if len(pycontrol_times) > len(tracking_df):  # sometimes sleap output is missing from end of session
        pycontrol_times = pycontrol_times[: len(tracking_df)]
    elif len(tracking_df) > len(pycontrol_times):  # catch cases where sync pulses stop early
        pycontrol_times = _extend_times(pycontrol_times, len(tracking_df))
    # get head direction
    head_direction, hd_interpolated = hd.get_head_direction(tracking_df)
    trajectories_df = pd.DataFrame(
        {
            ("time"): pycontrol_times,
            ("head_direction", "value"): head_direction,
            ("head_direction", "interpolated"): hd_interpolated,
            ("centroid_position", "x"): centroid.x,
            ("centroid_position", "y"): centroid.y,
            ("centroid_position", "interpolated"): centroid_interpolated,
        }
    )
    # create a MultiIndex for the column names
    columns = pd.MultiIndex.from_tuples(
        [(col, "") if isinstance(col, str) else col for col in trajectories_df.columns],
    )
    trajectories_df.columns = columns
    return trajectories_df


def _extend_times(pycontrol_times, final_length):
    """
    In some unlucky instances the sync pulses file stops early and as a result
    pycontrol times stop early, we can correct with with a linear extrapolation to some
    final array length
    """
    delta = np.mean(np.diff(pycontrol_times))
    array_len_diff = final_length - len(pycontrol_times)
    extra_times = pycontrol_times[-1] + delta * np.arange(1, array_len_diff + 1)
    return np.concatenate([pycontrol_times, extra_times])


def remove_far_away_points(tracking_df):
    """Remove outliers in the input DataFrame by setting points farther than a threshold from the centroid to NaN.
    Also removes points outside of the maze."""
    median_x = tracking_df.xs("x", axis=1, level=1).median(axis=1)
    median_y = tracking_df.xs("y", axis=1, level=1).median(axis=1)
    centroids = np.stack((median_x, median_y), axis=1)
    for part in tracking_df.columns.get_level_values(0).unique():
        ## Remove body parts that are far away from the centroid
        part_coords = tracking_df[[(part, "x"), (part, "y")]].to_numpy()
        distances = np.linalg.norm(part_coords - centroids, axis=1)
        qc_threshold = BODYPART_OUTLIER_THRESHOLDS[part]
        qc_mask = distances > qc_threshold
        part_coords[qc_mask] = np.nan
        ## Further remove body parts that are far away from the maze
        maze_outer_bound = (
            MAZE_MEASUREMENTS["tower_width"]
            + (MAZE_MEASUREMENTS["maze_node_dimensions"][0] - 1) * MAZE_MEASUREMENTS["distance_between_node_centers"]
            + MAZE_MEASUREMENTS["lower_left_node_cartesian_center"][0]
            + 0.01
        )  # add one for good measure.
        outside_maze = (
            (part_coords[:, 0] < 0)
            | (part_coords[:, 0] > maze_outer_bound)
            | (part_coords[:, 1] < 0)
            | (part_coords[:, 1] > maze_outer_bound)
        )
        part_coords[outside_maze] = np.nan
        tracking_df[[(part, "x"), (part, "y")]] = part_coords
    return tracking_df


def remove_flips(tracking_df, threshold=100):
    """
    Fixes errors in SLEAP output where the head is confused with the tail.
    This is most easily detectable as a large change in head direction.
    The logic follows the strategy of remove_jump_points but on the head direction data.
        1. Find longest stable segment
        2. Perform forward pass from the baseline (and backwards pass) to mark faulty indices
        3. Map faulty indices back to original series and set to NaN
    """
    df = tracking_df.copy()
    for col in df.columns.get_level_values(0).unique():
        positions = df[col]
        arr = positions.values  # 2D numpy array
        # Find indices where data is valid
        valid_mask = ~(df.head_back.x.isna() | df.head_back.y.isna())
        valid_idx = np.where(valid_mask == 1)[0]
        if sum(valid_mask) == 0:
            continue  # No valid data in this column
        valid_arr = arr[valid_mask]
    return df


def remove_jump_points(tracking_df, threshold=0.08, max_frames=30):
    """
    For each column in the DataFrame dlc_df, remove (set to NaN) frames that are part of
    persistent jump blocks. The algorithm works on the non-NaN values of each column:
      1. It finds the longest stable segment (with small frame-to-frame differences) and
         picks its midpoint as a baseline.
      2. It then performs a forward pass (and a backward pass via reversing the data)
         from the baseline, marking as faulty any indices where the difference from the last
         valid value exceeds the threshold.
      3. Faulty indices (from either pass) are mapped back to the original series and set to NaN.

    Parameters
    ---------
    dlc_df: pandas dataframe
        multiindexed with ("body_part","x")  and ("body_part","y") data.
    threshold: int()
        difference in metres considered to be a jump, 0.08 being equivalent to 4.8m/s speed at 60fps.
    max_frames: int()
        maximums number of frames for a segment of consecutive faulty positions.
    Returns:
        pd.DataFrame with faulty frames set to NaN.
    """

    df = tracking_df.copy()
    for col in df.columns.get_level_values(0).unique():
        positions = df[col]
        arr = positions.values  # 2D numpy array
        # Find indices where data is valid
        valid_mask = ~(df.head_back.x.isna() | df.head_back.y.isna())
        valid_idx = np.where(valid_mask == 1)[0]
        if sum(valid_mask) == 0:
            continue  # No valid data in this column
        valid_arr = arr[valid_mask]

        # Identify the longest stable segment among the valid data.
        seg_start, seg_end = _find_longest_stable_segment(valid_arr, threshold)
        baseline_idx = (seg_start + seg_end) // 2  # index in valid_arr
        # Run forward and backward error detection.
        faults = _find_jumps(valid_arr, baseline_idx, threshold, max_frames)
        # Map these fault flags back to the original array indices.
        faulty_indices = valid_idx[faults]
        arr[faulty_indices] = np.nan
        df[col] = arr
    return df


def _find_longest_stable_segment(valid_arr, threshold, modulus=10000):
    """
    Find the longest contiguous segment in a 2D numpy array (valid_arr) where
    the difference in distance between consecutive frames (speed) is ≤ threshold.

    Returns:
        (start, end): the indices (in valid_arr) of the longest stable segment.
    """
    if len(valid_arr) == 0:
        return 0, 0
    if len(valid_arr.shape) == 1:
        speed = np.abs(np.diff(valid_arr)) % modulus
    elif len(valid_arr.shape) == 2:
        speed = np.linalg.norm(np.diff(valid_arr, axis=0), axis=1) % modulus
    stable = speed <= threshold  # Boolean array of length len(valid_arr)-1

    best_start, best_end = 0, 0
    current_start = 0
    best_len = 1  # At minimum, a single element is "stable"
    for i in range(1, len(valid_arr)):
        if stable[i - 1]:
            # Continue the current segment
            current_len = i - current_start + 1
        else:
            current_len = i - current_start
            if current_len > best_len:
                best_len = current_len
                best_start, best_end = current_start, i - 1
            current_start = i
    # Check the last segment
    current_len = len(valid_arr) - current_start
    if current_len > best_len:
        best_start, best_end = current_start, len(valid_arr) - 1
    return best_start, best_end


def _find_jumps(valid_arr, baseline_idx, threshold, max_frames, modulus=10000):
    """
    Starting at baseline_idx in valid_arr, mark subsequent indices as faulty if the
    difference from the current "good" value exceeds threshold.
    """
    forward_faults = np.zeros(len(valid_arr), dtype=bool)
    last_valid = valid_arr[baseline_idx]
    last_valid_idx = baseline_idx
    for i in range(baseline_idx + 1, len(valid_arr)):
        measure = np.linalg.norm(valid_arr[i] - last_valid) % modulus
        if (measure > threshold) and (i - last_valid_idx < max_frames):
            forward_faults[i] = True
        else:
            last_valid = valid_arr[i]
            last_valid_idx = i

    backward_faults = np.zeros(len(valid_arr), dtype=bool)
    last_valid = valid_arr[baseline_idx]
    last_valid_idx = baseline_idx
    for i in range(baseline_idx - 1, -1, -1):
        measure = np.linalg.norm(valid_arr[i] - last_valid) % modulus
        if (measure > threshold) and (last_valid_idx - i < max_frames):
            backward_faults[i] = True
        else:
            last_valid = valid_arr[i]
            last_valid_idx = i

    # Combine forward and backward faults
    faults = forward_faults | backward_faults
    return faults


# %% convert video to pycontrol times


def get_frame_pytimes(video_pinstate_filepath, pycontrol_filepath):
    """Get the pycontrol times corresponding to each frame in the video file."""
    pinstate_df = pd.read_csv(video_pinstate_filepath)  # 60fps
    video_sync_pulse_frames = pinstate_df[pinstate_df["GPIO1"] == 1].index.values[::3]
    pycontrol_sync_pulse_times = pydf.get_pycontrol_sync_times(pycontrol_filepath)
    pytime_videotime_aligner = rsync.Rsync_aligner(
        video_sync_pulse_frames,
        pycontrol_sync_pulse_times,
        units_A=1 / FRAME_RATE,
        units_B=1,
    )
    frame_times = pytime_videotime_aligner.A_to_B(pinstate_df.index.values, extrapolate=True)
    if any(np.isnan(frame_times)):
        frame_times = interpolate_missing_times(frame_times)
    return frame_times


def interpolate_missing_times(frame_times):
    missing_inds = np.isnan(frame_times)
    frame_times[missing_inds] = np.interp(
        np.flatnonzero(missing_inds), np.flatnonzero(~missing_inds), frame_times[~missing_inds]
    )
    return frame_times


# %%


def get_SLEAP_output_as_df(filepath):
    """Opens a SLEAP .h5 file as a pandas dataframe (x,y coordinates are in pixels with origin bottom left).
    Includes logic to deal with multiple instances of body parts in the same frame (occurs with labeled frames)."""
    with h5py.File(filepath, "r") as f:
        locations = f["tracks"][:].T  # [n_frames, n_body_parts, 2(x,y), n_instances]
        node_names = [n.decode() for n in f["node_names"][:]]
        n_instances = locations.shape[-1]
    sleap_df = pd.DataFrame()
    for i, body_part in enumerate(node_names):
        if not n_instances > 1:
            sleap_df[(body_part, "x")] = locations[:, i, 0, 0]
            sleap_df[(body_part, "y")] = IMAGE_SIZE[0] - locations[:, i, 1, 0]
        else:  # deal with cases of multiple instances (happens is sessions with labeled frames)
            instance_dfs = []
            for instance in range(locations.shape[-1]):
                instance_df = pd.DataFrame()
                instance_df["x"] = locations[:, i, 0, instance]
                instance_df["y"] = IMAGE_SIZE[0] - locations[:, i, 1, instance]
                instance_dfs.append(instance_df)
            instance_dfs = sorted(instance_dfs, key=lambda x: x.isna().sum().sum())  # sort by number of nans
            base_df = instance_dfs[0].copy()
            for df in instance_dfs[1:]:
                base_df.update(df.copy())  # updates NaN values in base_df with values from instance_dfs
            sleap_df[(body_part, "x")] = base_df.x
            sleap_df[(body_part, "y")] = base_df.y
        sleap_df.columns = pd.MultiIndex.from_tuples(sleap_df.columns)
    # REORDER THE COLUMNS : IMPORTANT FOR HEAD_DIRECTION
    sleap_df = sleap_df[
        ["head_front", "head_mid", "head_back", "ear_L", "ear_R", "body_front", "body_mid", "body_back"]
    ]

    return sleap_df


# %% Choose outlier thresholds
def save_bodypart_outlier_thresholds():
    bodypart_outlier_thresholds = get_bodypart_outlier_thresholds()
    with open(Path(SLEAP_PATH) / "bodypart_outlier_thresholds.json", "w") as fp:
        json.dump(bodypart_outlier_thresholds, fp, indent=4)


def get_bodypart_outlier_thresholds():
    """Defines the outlier threshold for each body part as 5x the estimated standard deviation (sd) of the distance between body parts and body center.
    sd is estimated as the interquartile range of the distribution of distances between each body part and the body center (median position of all body parts),
    over all frames from a representative set of videos."""
    filepaths = get_representative_SLEAP_filepaths()
    distances_dfs = pd.DataFrame()
    for file in filepaths:
        sleap_df = get_SLEAP_output_as_df(file)
        sleap_df = translate_to_physical_coords(sleap_df)
        median_x = sleap_df.xs("x", axis=1, level=1).median(axis=1)
        median_y = sleap_df.xs("y", axis=1, level=1).median(axis=1)
        centroids = np.stack((median_x, median_y), axis=1)
        part2distances = {}
        for part in sleap_df.columns.get_level_values(0).unique():
            part_coords = sleap_df[[(part, "x"), (part, "y")]].to_numpy()
            distances = np.linalg.norm(part_coords - centroids, axis=1)
            part2distances[part] = distances
        distances_df = pd.DataFrame(part2distances)
        distances_dfs = pd.concat((distances_dfs, distances_df), axis=0, ignore_index=True)
    sd = (distances_df.quantile(0.886) - distances_df.quantile(0.114)) / 2
    threshold_multiplier = 5
    threshold_distances = distances_df.median() + sd * threshold_multiplier
    return dict(threshold_distances)


def get_representative_SLEAP_filepaths(session_types=["maze", "open_field"]):
    """Get first and last sleap tracking session for each subject from both maze and open field session types."""
    representative_filepaths = []
    all_sleap_paths = list(Path(SLEAP_PATH).iterdir())
    for subject in SUBJECT_IDS:
        subject_sleap_paths = [p for p in all_sleap_paths if subject in p.name]
        for session_type in session_types:
            session_paths = [p for p in subject_sleap_paths if session_type in p.name]
            session_datetimes = [  # convert datetime enbedded in filename to datetime object
                dt.strptime(p.name.split(".")[1].split("_")[-1], "%Y-%m-%d-%H%M%S") for p in session_paths
            ]
            representative_filepaths.append(session_paths[np.argmin(session_datetimes)])
            representative_filepaths.append(session_paths[np.argmax(session_datetimes)])
    return representative_filepaths
