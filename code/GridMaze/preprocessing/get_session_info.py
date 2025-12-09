"""
This script gets session information from pycontrol files using details manually specified in experiment_info folder.
Note: needs to be updated to deal with object vector sessions.

Author: @peterdoohan
"""

# %% Imports
import json
import numpy as np
from pathlib import Path
from GridMaze.preprocessing import get_pycontrol_dfs as pydf

# %% Global variables
EXPERIEMENT_INFO_PATH = Path("../data/experiment_info")

with open(EXPERIEMENT_INFO_PATH / "maze_info.json", "r") as infile:
    MAZE_INFO = json.load(infile)


# %% New Main functions


def get_session_info(session_dir):
    """ """
    session_df = pydf.load_session_dataframe(session_dir.pycontrol_path)
    maze_name = session_dir.maze_name
    maze_info = MAZE_INFO[maze_name]
    session_info = {
        "subject_ID": session_dir.subject_ID,
        "condition": session_dir.condition,
        "sex": session_dir.sex,
        "session_type": session_dir.session_type,
        "tethered": bool(session_dir.tethered),
        "experiment_phase": session_dir.experiment_phase,
        "maze_name": maze_name,
        "day_on_maze": int(session_dir.day_on_maze),
        "stim": bool(session_dir.stim),
        "stim_day": None if np.isnan(session_dir.stim_day) else int(session_dir.stim_day),
        "total_stim_days": None if np.isnan(session_dir.total_stim_days) else int(session_dir.total_stim_days),
        "date": session_dir.date.isoformat(),
        "experimental_day": int(session_dir.experimental_day),
        "maze_structure": maze_info["structure"],
        "goal_set": maze_info["goals"],
        "big_maze_rig": session_dir.big_maze_rig,
        "reward_size": _get_reward_vol(session_df),
    }
    return session_info


# %% Subfunctions


def _get_reward_vol(session_df):
    """Returns reward volumne in uL from pycontrol output"""
    return session_df[session_df.subtype == "run_start"].content.to_list()[0]["reward_vol_ul"]
