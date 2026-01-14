"""
Quant of errors during navigation
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import FancyArrowPatch, Circle

from GridMaze.analysis.core import get_sessions as gs
from GridMaze.maze import plotting as mp
from GridMaze.maze import representations as mr

# %% Global Variables

MAX_STIM_DURATION = 30

from GridMaze.paths import EXPERIMENT_INFO_PATH, RESULTS_PATH

with (EXPERIMENT_INFO_PATH / "subject_IDs.json").open("r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% eogcentric error map functions
