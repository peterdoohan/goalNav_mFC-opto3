"""
Script for converting raw_data & preprocessed_data from GridMaze experiments (organised in a session_data_directory dataframe)
into the GridMaze standardised preprocessed format, saved in the processed_data folder

Author: @peterdoohan
"""

# %% Imports
import json
from pathlib import Path

from GridMaze.preprocessing import get_frames_dfs as fd
from GridMaze.preprocessing.get_session_info import get_session_info
from GridMaze.preprocessing.get_pycontrol_dfs import get_events_df, get_trials_df
from GridMaze.preprocessing.get_data_directory import get_sessions_data_directory
from GridMaze.preprocessing import get_anatomy_data as gad


# %% Global Variables
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")
with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

SESSIONS_DATA_DIRECTORY = get_sessions_data_directory()
PROCESSED_DATA_PATH = Path("../data/processed_data")

if not PROCESSED_DATA_PATH.exists():  # check processed data folders exist
    PROCESSED_DATA_PATH.mkdir(parents=True)
for subject in SUBJECT_IDS:
    if not (PROCESSED_DATA_PATH / subject).exists():
        (PROCESSED_DATA_PATH / subject).mkdir(parents=True)

# %% Main functions


def populate_processed_data(
    session_data_streams=["pycontrol", "session_info", "video"],
    populate_anatomy_data=True,
    anatomy_data_structures=["registered_signal", "fiber_coordinates", "anatomy_info"],
    subject_IDs="all",
    session_dates="all",
    overwrite=False,
):
    """
    Top level function for populating standardised processed data.

    Args:
    - session_data_streams: list of raw(+ preprocessed) data streams to be processed at the session level.
        Options are ["pycontrol", "session_info", "video"]
    - subjects: list of subject IDs to process. Default is 'all', subjects in the SESSIONS_DATA_DIRECTORY dataframe
    - session_dates: list of session dates (datetime.date objects) to process. Default is 'all' session dates in
        the session_data_directory dataframe

    Notes:
    - see preprocessing module README.md for information about the processed data files populated for each data stream.
    """

    data_stream2func = {
        "pycontrol": populate_processed_pycontrol_data,
        "session_info": populate_processed_session_info,
        "video": populate_processed_video_data,
    }

    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    sessions_data_directory = SESSIONS_DATA_DIRECTORY

    # filter sessions
    if not session_dates == "all":
        sessions_data_directory = sessions_data_directory[sessions_data_directory.date.isin(session_dates)]
    if not subject_IDs == SUBJECT_IDS:
        sessions_data_directory = sessions_data_directory[sessions_data_directory.subject_ID.isin(subject_IDs)]
    # process session level data
    failed_sessions = []
    for data_stream, func in data_stream2func.items():
        if not data_stream in session_data_streams:
            continue
        else:
            print(f"Processing {data_stream} data")
            for session_dir in sessions_data_directory.itertuples():
                if not isinstance(session_dir.pycontrol_path, str):
                    print(f"No raw data yet for {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}")
                    continue
                processed_data_path = (
                    PROCESSED_DATA_PATH
                    / session_dir.subject_ID
                    / (session_dir.date.isoformat() + "." + session_dir.session_type)
                )
                if not processed_data_path.exists():
                    processed_data_path.mkdir()
                print(f"Processing {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}")
                try:
                    func(session_dir, processed_data_path, overwrite)
                except Exception as e:
                    print(
                        f"Error processing {data_stream} data for {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}: {e}"
                    )
                    failed_sessions.append(
                        f"{session_dir.subject_ID} {session_dir.date} {session_dir.session_type} {data_stream}"
                    )
    if len(failed_sessions) > 0:
        print("The following sessions failed to process:")
        for session in failed_sessions:
            print(session)

    if populate_anatomy_data:
        print("Processing anatomy data")
        gad.process_anatomy_data(anatomy_data_structures, overwrite=overwrite, verbose=True)

    return print("Finished populating processed data")


# %% session level data


def populate_processed_session_info(session_dir, processed_data_path, overwrite):
    """Generates and saves session_info.json to the processed_data folder for a given session."""
    if not isinstance(session_dir.pycontrol_path, str):  # nan or none
        print(
            f"pycontrol data not found for {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}, cannot populate processed session info data"
        )
        return
    else:
        filepath = processed_data_path / "session_info.json"
        if not _data_exists(filepath, overwrite):
            session_info = get_session_info(session_dir)
            with open(processed_data_path / "session_info.json", "w") as outfile:
                outfile.write(json.dumps(session_info, indent=4))
        return


def populate_processed_pycontrol_data(session_dir, processed_data_path, overwrite):
    """Generates and saves trials.htsv and events.htsv to the processed_data folder for a given session."""
    if not isinstance(session_dir.pycontrol_path, str):  # nan or none
        print(
            f"pycontrol data not found for {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}, cannot populate processed pycontrol data"
        )
        return
    else:
        # process events df
        events_df_path = processed_data_path / "events.htsv"
        if not _data_exists(events_df_path, overwrite):
            events_df = get_events_df(session_dir)
            events_df.to_csv(processed_data_path / "events.htsv", index=False, sep="\t")
        trials_df_path = processed_data_path / "trials.htsv"
        if not _data_exists(trials_df_path, overwrite):
            trials_df = get_trials_df(session_dir)
            # flatten multiindex columns to save as .htsv
            trials_df.columns = get_flattered_multiindex_columns(trials_df)
            trials_df.to_csv(processed_data_path / "trials.htsv", index=False, sep="\t")
        return


def populate_processed_video_data(session_dir, processed_data_path, overwrite):
    """Generates and saves frames.tracking.htsv for open_field sessions + addition frames.trajectories.htsv
    and frames.trialInfo.htsv files for maze sessions to the processed_data folder for a given session."""
    if not all(
        [
            isinstance(x, str)
            for x in [
                session_dir.video_path,
                session_dir.pycontrol_path,
                session_dir.video_sync_path,
                session_dir.SLEAP_path,
            ]
        ]
    ):
        print(
            f"video/sync/pycontrol/SLEAP data not found for {session_dir.subject_ID} {session_dir.date} {session_dir.session_type}, cannot populate processed video data"
        )
        return
    else:
        tracking_df_path = processed_data_path / "frames.tracking.htsv"
        if not _data_exists(tracking_df_path, overwrite):
            tracking_df = fd.get_tracking_df(session_dir)
            tracking_df.columns = get_flattered_multiindex_columns(tracking_df)
            tracking_df.to_csv(processed_data_path / "frames.tracking.htsv", index=False, sep="\t")
        trajectories_df_path = processed_data_path / "frames.trajectories.htsv"
        if not _data_exists(trajectories_df_path, overwrite):
            trajectories_df = fd.get_trajectories_df(session_dir)
            trajectories_df.columns = get_flattered_multiindex_columns(trajectories_df)
            trajectories_df.to_csv(processed_data_path / "frames.trajectories.htsv", index=False, sep="\t")
        trial_info_df_path = processed_data_path / "frames.trialInfo.htsv"
        if not _data_exists(trial_info_df_path, overwrite):
            trial_info_df = fd.get_trial_info_df(session_dir)
            trial_info_df.to_csv(processed_data_path / "frames.trialInfo.htsv", index=False, sep="\t")
        return


# %% Sub functions


def _data_exists(filepath, overwrite):
    if filepath.exists() and not overwrite:
        return True
    return False


def get_flattered_multiindex_columns(df):
    """Returns a list of flat column names (str) where columns that were previously multiindex become level0_name.level1_name
    and single index columns stay level0_name"""
    return [f"{x[0]}.{x[1]}" if x[1] != "" else x[0] for x in df.columns.to_flat_index()]
