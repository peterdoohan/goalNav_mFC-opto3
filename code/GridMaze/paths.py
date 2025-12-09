"""
Module for loadining in global variables common to many analyses / preprocessing steps
"""

# Imports
import json
import pandas as pd
from pathlib import Path

# %% Define exp paths
EXPERIMENT_INFO_PATH = Path("../data/experiment_info")

RAW_DATA_PATH = Path("../data/raw_data")

PREPROCESSED_DATA_PATH = Path("../data/preprocessed_data")

PYCONTROL_PATH = RAW_DATA_PATH / "pycontrol"

VIDEO_PATH = RAW_DATA_PATH / "video"

SLEAP_PATH = PREPROCESSED_DATA_PATH / "SLEAP"

PROCESSED_DATA_PATH = Path("../data/processed_data")
ANALYSIS_DATA_PATH = Path("../data/analysis_data")

ANALYSIS_INFO_PATH = Path("../data/analysis_data/analysis_info")

RESULTS_PATH = Path("../results")
