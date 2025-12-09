"""
Library to house functions and classes for organizing processed and anlysis data into session objects, that are the
starting point for any analysis
@peterdoohan
"""

# %% Imports

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

from GridMaze.analysis.core import load_data
from GridMaze.maze import representations as mr

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH, PROCESSED_DATA_PATH, ANALYSIS_DATA_PATH

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)


PROCESSED_DATA_STRUCTURE2FILENAME = {
    "session_info": "session_info.json",
    "trials_df": "trials.htsv",
    "events_df": "events.htsv",
    "tracking_df": "frames.tracking.htsv",
    "trajectories_df": "frames.trajectories.htsv",
    "trial_info_df": "frames.trialInfo.htsv",
}

ANALYSIS_DATA_STRUCTURE2FILENAME = {
    "navigation_df": "frames.navigation.parquet",
    "navigation_strategies_df": "navigation_strategies_dataframe.parquet",
    "trajectory_decisions_df": "trajectory_decisions_dataframe.parquet",
}

ALL_DATA_STRUCTURES2FILENAME = {**PROCESSED_DATA_STRUCTURE2FILENAME, **ANALYSIS_DATA_STRUCTURE2FILENAME}

# %% Main


def get_maze_sessions(
    subject_IDs="all",
    conditions="all",  # opto, control
    big_maze_rig="all",  # L, R
    sex="all",  # male, female
    maze_order="all",
    maze_names="all",  # maze_1, maze_2
    days_on_maze="all",
    experiment_phases="all",  # learning, expert
    stim_only=False,
    tethered_only=False,
    experimental_days="all",
    with_data="all",
    must_have_data=True,
    verbose=True,
):
    """ """
    # filter subjects based on input params
    subject_IDs = SUBJECT_IDS if subject_IDs == "all" else subject_IDs
    if not conditions == "all":
        subjects_in_condition = SUBJECT_INFO_DF[SUBJECT_INFO_DF.condition.isin(conditions)].subject_ID.to_list()
        subject_IDs = list(set(subjects_in_condition) & set(subject_IDs))
    if not big_maze_rig == "all":
        subjects_in_rig = SUBJECT_INFO_DF[SUBJECT_INFO_DF.big_maze_rig == big_maze_rig].subject_ID.to_list()
        subject_IDs = list(set(subject_IDs) & set(subjects_in_rig))
    if not sex == "all":
        subjects_of_sex = SUBJECT_INFO_DF[SUBJECT_INFO_DF.sex == sex].subject_ID.to_list()
        subject_IDs = list(set(subject_IDs) & set(subjects_of_sex))
    if len(subject_IDs) == 0:
        raise ValueError("No subjects found with the specified parameters")
    else:
        # filter data structures based on input params
        with_data = ALL_DATA_STRUCTURES2FILENAME.keys() if with_data == "all" else with_data
        # retrieve relevant sessions
        requested_sessions = []
        for subject_ID in subject_IDs:
            session_folders = list((PROCESSED_DATA_PATH / subject_ID).glob("*"))
            for session_folder in session_folders:
                if not session_folder.is_dir():
                    continue
                try:
                    with open(session_folder / "session_info.json", "r") as input_file:
                        session_info = json.load(input_file)
                except FileNotFoundError:
                    print(f"session_info.json not found for {session_folder}")
                    continue
                if stim_only and not session_info["stim"]:
                    continue
                if tethered_only and not session_info["tethered"]:
                    continue
                experimental_day = session_info["experimental_day"]
                if not experimental_days == "all":
                    if not experimental_day in experimental_days:
                        continue
                if not maze_order == "all":
                    if session_info["maze_order"] not in maze_order:
                        continue
                if not maze_names == "all":
                    if not session_info["maze_name"] in maze_names:
                        continue
                if not days_on_maze == "all":
                    if not session_info["day_on_maze"] in days_on_maze:
                        continue
                if not experiment_phases == "all":
                    if not session_info["experiment_phase"] in experiment_phases:
                        continue
                session = MazeSession(subject_ID, session_info["date"], with_data)
                if must_have_data:  # only add sessions that have all requested data
                    if all([getattr(session, data_structure) is not None for data_structure in with_data]):
                        requested_sessions.append(session)
                    else:
                        if verbose:
                            print(f"Session {session.name} does not have all requested data")
            if len(requested_sessions) == 0 and verbose:
                print("No sessions found with the specified parameters")
        return requested_sessions


class MazeSession:
    """ """

    def __init__(self, subject, session_date, with_data="all"):
        """ """
        self.has_data = with_data
        processed_data_path = PROCESSED_DATA_PATH / subject / f"{session_date}.maze"
        analysis_data_path = ANALYSIS_DATA_PATH / subject / f"{session_date}.maze"
        # Load session info
        with open(processed_data_path / "session_info.json", "r") as input_file:
            session_info = json.load(input_file)
        self.name = get_session_name(session_info)
        self.date = date.fromisoformat(session_info["date"])
        for attr_name in [k for k in session_info.keys() if k != "date"]:
            setattr(self, attr_name, session_info[attr_name])
        # Load processed data
        for attr_name, file_name in ALL_DATA_STRUCTURES2FILENAME.items():
            if with_data == "all" or attr_name in with_data:
                if attr_name in PROCESSED_DATA_STRUCTURE2FILENAME.keys():
                    file_path = processed_data_path / file_name
                elif attr_name in ANALYSIS_DATA_STRUCTURE2FILENAME.keys():
                    file_path = analysis_data_path / file_name
                try:
                    data = load_data.load(file_path)
                except FileNotFoundError:
                    print(f"{file_name} not found for {self.name}")
                    data = None
            else:
                data = None
            setattr(self, attr_name, data)
        return

    def __repr__(self):
        """Return a nicely formatted string representation of the object."""
        return (
            f"-MazeSession--------------------------------------------------------------\n"
            f"    Subject ID: {self.subject_ID}, Condition: {self.condition}, Date: {self.date}\n"
            f"    Maze: {self.maze_name}, Maze Order: {self.maze_order}, Day on Maze: {self.day_on_maze}, Stim: {self.stim}, \n"
        )

    def simple_maze(self):
        return mr.simple_maze(self.maze_structure)

    def skeleton_maze(self):
        return mr.skeleton_maze(self.maze_structure)


def get_session_name(session_info):
    return f"{session_info['subject_ID']}.{session_info['date']}.{session_info['session_type']}"
