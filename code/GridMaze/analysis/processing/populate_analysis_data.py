"""
Library for generating custom analysis data structures from standardised processed data, saving the outputs in
data/analysis_data.
@peterdoohan
"""

# %% Imports
import json
from pathlib import Path
from datetime import date
from joblib import Parallel, delayed

from GridMaze.analysis.processing.get_navigation_dfs import get_navigation_df
from GridMaze.analysis.processing.get_navigation_strategies_dfs import get_navigation_strategies_df
from GridMaze.analysis.processing.get_trajectory_decisions_dfs import get_trajectory_decisions_df

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, ANALYSIS_DATA_PATH, PROCESSED_DATA_PATH

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

# check analysis data folders exist
if not ANALYSIS_DATA_PATH.exists():
    ANALYSIS_DATA_PATH.mkdir(parents=True)
for subject in SUBJECT_IDS:
    if not (ANALYSIS_DATA_PATH / subject).exists():
        (ANALYSIS_DATA_PATH / subject).mkdir(parents=True)

# %% Parallel processing version


def populate_analysis_data(
    data_structures=[
        "navigation_df",
        "navigation_strategies_df",
        "trajectory_decisions_df",
    ],
    subject_IDs="all",
    overwrite=False,
    parallel_jobs=-1,
):
    """ """
    data_structure2func = {
        "navigation_df": _save_navigation_df,
        "navigation_strategies_df": _save_navigation_strategies_df,
        "trajectory_decisions_df": _save_trajectory_decisions_df,
    }
    if any([data_structure not in data_structure2func.keys() for data_structure in data_structures]):
        raise ValueError(f"Data structure not recognised, must be in {data_structure2func.keys()}")
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    processed_data_paths, analysis_data_paths = [], []
    for subject in subject_IDs:
        _processed_data_paths = [f for f in (PROCESSED_DATA_PATH / subject).iterdir() if f.is_dir()]
        processed_data_paths.extend(_processed_data_paths)
        analysis_data_paths.extend([ANALYSIS_DATA_PATH / subject / p.name for p in _processed_data_paths])

    def _process_session(processed_data_path, analysis_data_path):
        if not analysis_data_path.exists():
            analysis_data_path.mkdir(parents=True)
        print(f"Saving analysis data for {processed_data_path}")
        for data_structure in data_structures:
            try:
                data_function = data_structure2func[data_structure]
                data_function(processed_data_path, analysis_data_path, overwrite)
            except FileNotFoundError:
                print(f"FileNotFoundError: {data_function.__name__} failed for {processed_data_path}")
                pass

    if parallel_jobs:
        Parallel(n_jobs=parallel_jobs)(
            delayed(_process_session)(processed_data_path, analysis_data_path)
            for processed_data_path, analysis_data_path in zip(processed_data_paths, analysis_data_paths)
        )
    else:
        for processed_data_path, analysis_data_path in zip(processed_data_paths, analysis_data_paths):
            _process_session(processed_data_path, analysis_data_path)


# %% Functions


def _save_navigation_df(processed_data_path, analysis_data_path, overwrite):
    """ """
    prerequisit_data = [
        processed_data_path / "session_info.json",
        processed_data_path / "frames.trajectories.htsv",
        processed_data_path / "frames.trialInfo.htsv",
    ]
    if not _prerequisit_data_exists(prerequisit_data):  # check prequisit processed data structures exist
        return print(
            f"Missing pre-requisite processed data structures, cannot generate navigation_df for {processed_data_path.parts[-2:]}"
        )

    navigation_df_path = analysis_data_path / "frames.navigation.parquet"
    if not _data_exists(navigation_df_path, overwrite):
        navigation_df = get_navigation_df(processed_data_path)
        navigation_df.columns = navigation_df.columns.map(
            lambda x: str(x)
        )  # converts column names to strings for saving as parquet
        navigation_df.to_parquet(navigation_df_path, compression="gzip", index=False)
    return


def _save_navigation_strategies_df(processed_data_path, analysis_data_path, overwrite):
    """ """
    prerequisit_data = [
        processed_data_path / "trials.htsv",
        processed_data_path / "session_info.json",
        analysis_data_path / "frames.navigation.parquet",
    ]
    if not _prerequisit_data_exists(prerequisit_data):
        return print(
            f"Missing pre-requisite processed data structures, cannot generate navigation_strategies_df for {processed_data_path.parts[-2:]}"
        )
    navigation_strategies_df_path = analysis_data_path / "navigation_strategies_dataframe.parquet"
    if not _data_exists(navigation_strategies_df_path, overwrite):
        navigation_strategies_df = get_navigation_strategies_df(processed_data_path, analysis_data_path)
        navigation_strategies_df.columns = navigation_strategies_df.columns.map(lambda x: str(x))
        navigation_strategies_df.to_parquet(navigation_strategies_df_path, compression="gzip", index=False)
    return


def _save_trajectory_decisions_df(processed_data_path, analysis_data_path, overwrite):
    """ """
    prerequisit_data = [
        processed_data_path / "session_info.json",
        processed_data_path / "frames.trajectories.htsv",
        processed_data_path / "frames.trialInfo.htsv",
    ]
    if not _prerequisit_data_exists(prerequisit_data):
        return print(
            f"Missing pre-requisite processed data structures, cannot generate trajectory_decisions_df for {processed_data_path.parts[-2:]}"
        )

    trajectory_decisions_df_path = analysis_data_path / "trajectory_decisions_dataframe.parquet"
    if not _data_exists(trajectory_decisions_df_path, overwrite):
        trajectory_decisions_df = get_trajectory_decisions_df(processed_data_path)
        trajectory_decisions_df.columns = trajectory_decisions_df.columns.map(
            lambda x: str(x)
        )  # converts column names to strings for saving as parquet
        trajectory_decisions_df.to_parquet(trajectory_decisions_df_path, compression="gzip", index=False)
    return


# %% Supporting functions
def _prerequisit_data_exists(data_paths):
    if all([d.exists() for d in data_paths]):
        return True
    else:
        return False


def _data_exists(filepath, overwrite):
    if filepath.exists() and not overwrite:
        return True
    return False
