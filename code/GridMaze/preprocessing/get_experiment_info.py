"""
Creates multiple .json/ .htsv files with experimental details used for preprocessing and analysis,
saved in the data/experiment_info folder.
@peterdoohan
"""

# %% Imports
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

# %%

EXPERIMENT_INFO_PATH = Path("../data/experiment_info/")

# %% subject_IDs

SUBJECT_IDS = [
    "mFC-opto_23",
    "mFC-opto_24",
    "mFC-opto_25",
    "mFC-opto_26",
    "mFC-opto_27",
    "mFC-opto_28",
    "mFC-opto_29",
    "mFC-opto_30",
    "mFC-opto_31",
    "mFC-opto_32",
    "mFC-opto_34",
    "mFC-opto_35",
    "mFC-opto_36",
    "mFC-opto_37",
    "mFC-opto_38",
]

SUBJECT_INFO_DF = pd.DataFrame(
    [
        {
            "subject_ID": "mFC-opto_23",
            "sex": "male",
            "cage_name": "M1",
            "cage_color": "orange",
            "cap_marking": "orange",
            "condition": "opto",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 14),
        },
        {
            "subject_ID": "mFC-opto_24",
            "sex": "male",
            "cage_name": "M1",
            "cage_color": "orange",
            "cap_marking": "orange-orange",
            "condition": "control",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 14),
        },
        {
            "subject_ID": "mFC-opto_25",
            "sex": "male",
            "cage_name": "M1",
            "cage_color": "orange",
            "cap_marking": "orange-blue",
            "condition": "control",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 14),
        },
        {
            "subject_ID": "mFC-opto_26",
            "sex": "male",
            "cage_name": "M1",
            "cage_color": "orange",
            "cap_marking": "orange-black",
            "condition": "opto",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 14),
        },
        {
            "subject_ID": "mFC-opto_27",
            "sex": "male",
            "cage_name": "M2",
            "cage_color": "pink",
            "cap_marking": "pink",
            "condition": "opto",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 16),
        },
        {
            "subject_ID": "mFC-opto_28",
            "sex": "male",
            "cage_name": "M2",
            "cage_color": "pink",
            "cap_marking": "pink-pink",
            "condition": "control",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 16),
        },
        {
            "subject_ID": "mFC-opto_29",
            "sex": "male",
            "cage_name": "M2",
            "cage_color": "pink",
            "cap_marking": "pink-blue",
            "condition": "control",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 16),
        },
        {
            "subject_ID": "mFC-opto_30",
            "sex": "male",
            "cage_name": "M2",
            "cage_color": "pink",
            "cap_marking": "pink-black",
            "condition": "opto",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 16),
        },
        {
            "subject_ID": "mFC-opto_31",
            "sex": "female",
            "cage_name": "F1",
            "cage_color": "green",
            "cap_marking": "green",
            "condition": "opto",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 15),
        },
        {
            "subject_ID": "mFC-opto_32",
            "sex": "female",
            "cage_name": "F1",
            "cage_color": "green",
            "cap_marking": "green-green",
            "condition": "control",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 15),
        },
        {
            "subject_ID": "mFC-opto_34",
            "sex": "female",
            "cage_name": "F1",
            "cage_color": "green",
            "cap_marking": "green-black",
            "condition": "control",
            "big_maze_rig": "L",
            "surgery_date": date(2025, 10, 15),
        },
        {
            "subject_ID": "mFC-opto_35",
            "sex": "female",
            "cage_name": "F2",
            "cage_color": "yellow",
            "cap_marking": "yellow",
            "condition": "control",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 13),
        },
        {
            "subject_ID": "mFC-opto_36",
            "sex": "female",
            "cage_name": "F2",
            "cage_color": "yellow",
            "cap_marking": "yellow-yellow",
            "condition": "opto",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 13),
        },
        {
            "subject_ID": "mFC-opto_37",
            "sex": "female",
            "cage_name": "F2",
            "cage_color": "yellow",
            "cap_marking": "yellow-blue",
            "condition": "control",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 13),
        },
        {
            "subject_ID": "mFC-opto_38",
            "sex": "female",
            "cage_name": "F2",
            "cage_color": "yellow",
            "cap_marking": "yellow-black",
            "condition": "opto",
            "big_maze_rig": "R",
            "surgery_date": date(2025, 10, 13),
        },
    ]
)

IGNORE_SESSIONS = [
    {
        "subject_ID": "mFC-opto_30",
        "date": "2025-11-20",
        "reason": "pycontrol failed, session restarted, ignore for now, can be salvaged",
    },
    {
        "subject_ID": "mFC-opto_37",
        "date": "2025-11-23",
        "reason": "pycontrol failed, session restarted, ignore for now, can be salvaged",
    },
]

DAYS_OFF = ["2025-11-29"]

# %%

FIBER_TETHERED_DATE = date(
    2025, 11, 14
)  # subjects started maze learning untethered, then were fiber tethered from this date

# %% Maze structures specific by their connected edges
MAZE_1 = [
    "A1-A2",
    "A3-A4",
    "A4-A5",
    "A5-A6",
    "A6-A7",
    "A2-B2",
    "A3-B3",
    "A5-B5",
    "A7-B7",
    "B4-B5",
    "B6-B7",
    "B1-C1",
    "B2-C2",
    "B3-C3",
    "B6-C6",
    "C1-C2",
    "C2-C3",
    "C3-C4",
    "C4-C5",
    "C5-C6",
    "C6-C7",
    "C2-D2",
    "C5-D5",
    "C7-D7",
    "D1-D2",
    "D3-D4",
    "D4-D5",
    "D6-D7",
    "D1-E1",
    "D2-E2",
    "D3-E3",
    "D4-E4",
    "D5-E5",
    "D6-E6",
    "E2-F2",
    "E3-F3",
    "E5-F5",
    "E6-F6",
    "E7-F7",
    "F1-F2",
    "F2-F3",
    "F4-F5",
    "F6-F7",
    "F2-G2",
    "F5-G5",
    "F6-G6",
    "G1-G2",
    "G2-G3",
    "G3-G4",
    "G4-G5",
    "G5-G6",
    "G6-G7",
]

MAZE_2 = [
    "A1-A2",
    "A2-A3",
    "A3-A4",
    "A4-A5",
    "A5-A6",
    "A6-A7",
    "A1-B1",
    "A3-B3",
    "A5-B5",
    "A6-B6",
    "B1-B2",
    "B6-B7",
    "B1-C1",
    "B2-C2",
    "B4-C4",
    "B5-C5",
    "B7-C7",
    "C2-C3",
    "C3-C4",
    "C4-C5",
    "C5-C6",
    "C6-C7",
    "C1-D1",
    "C2-D2",
    "C5-D5",
    "D3-D4",
    "D4-D5",
    "D5-D6",
    "D6-D7",
    "D1-E1",
    "D3-E3",
    "D6-E6",
    "E1-E2",
    "E2-E3",
    "E4-E5",
    "E6-E7",
    "E3-F3",
    "E5-F5",
    "E7-F7",
    "F1-F2",
    "F2-F3",
    "F3-F4",
    "F4-F5",
    "F5-F6",
    "F6-F7",
    "F3-G3",
    "F5-G5",
    "F7-G7",
    "G1-G2",
    "G2-G3",
    "G4-G5",
    "G5-G6",
]

GOALS = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
]


MAZE_INFO = {"maze_1": {"structure": MAZE_1, "goals": GOALS}, "maze_2": {"structure": MAZE_2, "goals": GOALS}}

BIG_MAZE_RIG2MAZE_DATES = {
    "L": {
        "maze_1": {"start": "2025-11-10", "end": "2025-12-01", "start_stim": "2025-11-19"},
        "maze_2": {"start": "2025-12-02", "end": "2025-12-17", "start_stim": "2025-12-07"},
    },
    "R": {
        "maze_2": {"start": "2025-11-10", "end": "2025-12-01", "start_stim": "2025-11-19"},
        "maze_1": {"start": "2025-12-02", "end": "2025-12-17", "start_stim": "2025-12-07"},
    },
}

# %%  Maze measurements
MAZE_MEASUREMENTS = {
    "maze_node_dimensions": (7, 7),
    "lower_left_node_cartesian_center": (0.15, 0.15),  # meters
    "distance_between_node_centers": 0.18,  # meters
    "tower_width": 0.11,
}  # meters


# %% Functions

FILENAME2DATA_STRUCTURE = {
    "subject_IDs": SUBJECT_IDS,
    "fiber_tethered_date": FIBER_TETHERED_DATE.isoformat(),
    "maze_info": MAZE_INFO,
    "rig2maze_dates": BIG_MAZE_RIG2MAZE_DATES,
    "maze_measurements": MAZE_MEASUREMENTS,
    "ignore_sessions": IGNORE_SESSIONS,
    "days_off": DAYS_OFF,
}


def save_experiment_info():
    """
    Saves experimental details as .json files, or a .htsv file in the case of subject_info_df
    in specificed experiment_info folder
    """
    for filename, data_structure in FILENAME2DATA_STRUCTURE.items():
        with open(EXPERIMENT_INFO_PATH / (filename + ".json"), "w") as outfile:
            outfile.write(json.dumps(data_structure, indent=4))
    SUBJECT_INFO_DF.to_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t", index=False)
    return print(f"Experiment info saved to {EXPERIMENT_INFO_PATH}")
