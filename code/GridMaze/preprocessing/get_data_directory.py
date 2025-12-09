"""
This library houses a top-level function "get_raw_data_directory" (and supporting functions), that generates a pandas
DataFrame with a row for each session (of all session-types, eg. maze and openfield) and columns for all the paths
to all raw data files associated with that session (eg. pycontrol, video, SLEAP, ephys, etc.). Note that in GridMaze
experiments we consider data processed through standardised piplines (eg. SLEAP, Kilosort, etc.) to be "raw_data".
Currently, only sessions with all data easily available are selected for further preprocessing. In the future, sessions
with issues such as missing ephys data (record buffer error), restarted pycontrol files etc. can be included but this will
take custom code to deal with edge cases.

This DataFrame is later used in the populate_processed_data.py script where each row of the sessions_data_directory (pandas
Series) is used to access the raw data paths needed for each preprocessing step.
"""

# %% Imports
import json
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

# %% Global variables

from GridMaze.paths import (
    EXPERIMENT_INFO_PATH,
    PYCONTROL_PATH,
    VIDEO_PATH,
    SLEAP_PATH,
)

with open(EXPERIMENT_INFO_PATH / "start_date.json", "r") as infile:
    START_DATE = date.fromisoformat(json.load(infile))

with open(EXPERIMENT_INFO_PATH / "fiber_tethered_date.json", "r") as infile:
    FIBER_TETHERED_DATE = date.fromisoformat(json.load(infile))

with open(EXPERIMENT_INFO_PATH / "rig2maze_dates.json", "r") as infile:
    RIG2MAZE_DATES = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "ignore_sessions.json", "r") as infile:
    IGNORE_SESSIONS = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "session_notes.json", "r") as infile:
    SESSION_NOTES = json.load(infile)

with open(EXPERIMENT_INFO_PATH / "days_off.json", "r") as infile:
    DAYS_OFF = json.load(infile)

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")


FRAME_RATE = 60  # Hz
# %% new


def get_sessions_data_directory():
    """ """
    sessions_data_directory = init_data_directory()
    sessions_data_directory["pycontrol_path"] = _add_pycontrol_paths(sessions_data_directory, get_pycontrol_paths_df())
    video_paths, video_sync_paths = _add_video_paths(sessions_data_directory, get_video_paths_df())
    sessions_data_directory["video_path"] = video_paths
    sessions_data_directory["video_sync_path"] = video_sync_paths
    sessions_data_directory["SLEAP_path"] = _add_SLEAP_paths(sessions_data_directory, get_SLEAP_paths_df())
    return sessions_data_directory


def init_data_directory():
    """ """
    _subject_info_df = SUBJECT_INFO_DF.set_index("subject_ID")
    today_date = date.today()
    session_infos = []
    for subject_ID in SUBJECT_IDS:
        subject_info = _subject_info_df.loc[subject_ID]
        big_maze_rig = subject_info["big_maze_rig"]
        total_stim_days = 0
        for i, (maze_name, _dates) in enumerate(RIG2MAZE_DATES[big_maze_rig].items()):
            start_date = date.fromisoformat(_dates["start"])
            end_date = date.fromisoformat(_dates["end"])
            stim_start_date = date.fromisoformat(_dates["start_stim"])
            day_on_maze = 1
            if end_date > today_date:
                end_date = today_date
            _date = start_date
            while _date <= end_date:
                exp_day = (_date - START_DATE).days + 1
                tethered = True if _date >= FIBER_TETHERED_DATE else False
                if _date >= stim_start_date:
                    stim = True
                    total_stim_days += 1
                    stim_day = (_date - stim_start_date).days + 1
                    exp_phase = "expert"
                else:
                    stim = False
                    stim_day = np.nan
                    exp_phase = "learning"
                if _check_ignore(subject_ID, _date):
                    _date += timedelta(days=1)
                    day_on_maze += 1
                    continue
                session_notes = _get_session_notes(subject_ID, _date)
                session_info = {
                    "subject_ID": subject_ID,
                    "condition": subject_info.condition,
                    "sex": subject_info.sex,
                    "date": _date,
                    "experimental_day": exp_day,
                    "session_type": "maze",
                    "big_maze_rig": subject_info.big_maze_rig,
                    "maze_name": maze_name,
                    "maze_order": i + 1,
                    "day_on_maze": day_on_maze,
                    "experiment_phase": exp_phase,
                    "tethered": tethered,
                    "stim": stim,
                    "stim_day": stim_day,
                    "total_stim_days": total_stim_days if stim else np.nan,
                    "session_notes": session_notes,
                }
                session_infos.append(session_info)
                _date += timedelta(days=1)
                day_on_maze += 1
    return pd.DataFrame(session_infos)


def _get_session_notes(subject_ID, date):
    for session_note in SESSION_NOTES:
        if session_note["subject_ID"] == subject_ID and session_note["date"] == date.isoformat():
            return session_note["notes"]
    return None


def _check_ignore(subject_ID, date):
    if date.isoformat() in DAYS_OFF:
        return True
    for ig in IGNORE_SESSIONS:
        if ig["subject_ID"] == subject_ID and ig["date"] == date.isoformat():
            return True
    return False


# %% raw_data: pycontrol
def get_pycontrol_paths_df():
    """
    extracts the subject, and datetime from the experiment pycontrol files and organises
    them into a pd.DataFrame.
    """
    all_pycontrol_paths = [pf.name for pf in Path(PYCONTROL_PATH).glob("*.tsv") if not pf.name.startswith(".")]
    # handle ignore sessions here
    pycontrol_path_info = []
    for path in all_pycontrol_paths:
        pycontrol_path_info.append(
            {
                "subject_ID": "-".join(path.split("-")[:2]),
                "datetime": datetime.strptime(path.split("-", 2)[-1].split(".")[0], "%Y-%m-%d-%H%M%S"),
                "pycontrol_path": str(Path(PYCONTROL_PATH) / path),
            }
        )
    return pd.DataFrame(pycontrol_path_info)


def _add_pycontrol_paths(init_data_directory, pycontrol_paths_df):
    """
    Returns the path to the pycontrol file associated with each session in the init_data_directory, ordered
    by sessions in the session_data_directory DataFrame.
    """
    paths = []
    for row in init_data_directory.itertuples():
        filtered_sessions = pycontrol_paths_df[
            np.logical_and.reduce(
                [
                    (pycontrol_paths_df.subject_ID == row.subject_ID),
                    (pycontrol_paths_df.datetime.apply(datetime.date) == row.date),
                ]
            )
        ]
        assert len(filtered_sessions) <= 1, f"Multiple pycontrol files found for for {row.subject_ID} on {row.date}"
        # if no pycontrol file found, add np.nan
        if len(filtered_sessions) == 0:
            print(f"no pycontrol file found for {row.subject_ID} on {row.date}")
            paths.append(np.nan)
        else:
            paths.append(filtered_sessions.pycontrol_path.values[0])
    return paths


def _get_pycontrol_session_duration(pycontrol_path):
    """Returns the length of a pycontrol session (in minutes) from the pycontrol file input"""
    if pycontrol_path is np.nan:
        return np.nan
    else:
        raw_pycontrol_df = pd.read_csv(pycontrol_path, sep="\t")
        return raw_pycontrol_df.time.iloc[-1] / 60


# %% raw_data: video


def _add_video_paths(init_data_directory, video_paths_df):
    video_paths, video_sync_paths = [], []
    for row in init_data_directory.itertuples():
        filtered_sessions = video_paths_df[
            np.logical_and.reduce(
                [
                    (video_paths_df.subject_ID == row.subject_ID),
                    (video_paths_df.datetime.apply(datetime.date) == row.date),
                ]
            )
        ]
        if len(filtered_sessions) > 1:
            assert f"Multiple video files found for for {row.subject_ID}, {row.date}"
        # if no video file found, add np.nan
        if len(filtered_sessions) == 0:
            print(f"no video file found for {row.subject_ID} on {row.date}")
            video_paths.append(np.nan)
            video_sync_paths.append(np.nan)
        else:
            video_paths.append(filtered_sessions.video_path.values[0])
            video_sync_paths.append(filtered_sessions.video_sync_path.values[0])
    return video_paths, video_sync_paths


def get_video_paths_df():
    """
    Returns a pd.Dataframe with data extracted from video filenames,
    (rows: sessions, columns: datetime, video_path, vidoe_sync_pulse_path).
    Will later match to pycontrol sessions with nearest datetime. Note some errors in
    the video filenames (wrong subject, wrong session type etc. havn't been fixed in the video files
    but datetimes should always line up so that is all we need).

    Notes
     - session_types for obj vect sessions are just open_field in video filenames
     - no video for rest sessions
    """
    all_video_files = [f.name for f in Path(VIDEO_PATH).iterdir() if f.suffix == ".mp4"]
    all_sync_pulse_files = [f.name for f in Path(VIDEO_PATH).iterdir() if f.suffix == ".csv"]
    video_paths_info = []
    for video_file in all_video_files:
        video_paths_info.append(
            {
                "subject_ID": "_".join(video_file.split("_")[:2]),
                "datetime": datetime.strptime(video_file.split("_")[-1].split(".")[0], "%Y-%m-%d-%H%M%S"),
                "video_path": str(Path(VIDEO_PATH) / video_file),
            }
        )
    video_paths_df = pd.DataFrame(video_paths_info)
    sync_pulse_info = []
    for sync_file in all_sync_pulse_files:
        sync_pulse_info.append(
            {
                "datetime": datetime.strptime(sync_file.split("_")[2], "%Y-%m-%d-%H%M%S"),
                "video_sync_path": str(Path(VIDEO_PATH) / sync_file),
            }
        )
    sync_pulse_df = pd.DataFrame(sync_pulse_info)
    # merge dfs
    merged_video_paths_df = pd.merge(video_paths_df, sync_pulse_df, on="datetime", how="inner")
    return merged_video_paths_df


# %% finding SLEAP data


def _add_SLEAP_paths(init_data_directory, SLEAP_paths_df):
    sleap_paths = []
    for row in init_data_directory.itertuples():
        filtered_sessions = SLEAP_paths_df[
            np.logical_and.reduce(
                [
                    (SLEAP_paths_df.subject_ID == row.subject_ID),
                    (SLEAP_paths_df.datetime.apply(datetime.date) == row.date),
                ]
            )
        ]
        if len(filtered_sessions) > 1:
            print(f"Multiple SLEAP files found fo {row.subject_ID}, {row.date}")
        # if no SLEAP file found, add np.nan
        if len(filtered_sessions) == 0:
            print(f"no SLEAP file found for {row.subject_ID} on {row.date}")
            sleap_paths.append(np.nan)
        else:
            sleap_paths.append(filtered_sessions.SLEAP_path.values[0])
    return sleap_paths


def get_SLEAP_paths_df():
    all_sleap_files = [f.name for f in Path(SLEAP_PATH).iterdir() if f.suffix == ".h5"]
    sleap_path_info = []
    for sleap_file in all_sleap_files:
        sleap_path_info.append(
            {
                "subject_ID": "_".join(sleap_file.split("_")[:2]),
                "datetime": datetime.strptime(sleap_file.split(".")[0].split("_")[-1], "%Y-%m-%d-%H%M%S"),
                "SLEAP_path": str(Path(SLEAP_PATH) / sleap_file),
            }
        )
    return pd.DataFrame(sleap_path_info)


def _get_video_duration(SLEAP_path):
    """Calculates Video length from SLEAP data, more efficent than loading video files."""
    if SLEAP_path is np.nan:
        return np.nan
    else:
        try:
            with h5py.File(SLEAP_path, "r") as f:
                data = f["tracks"][:].T
            duration = data.shape[0]  # frames
            duration = duration / FRAME_RATE  # seconds
            duration = duration / 60  # minutes
        except OSError:
            raise f"Error reading SLEAP file: {SLEAP_path}"
        return duration
