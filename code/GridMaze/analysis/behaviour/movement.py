"""
Lib for quantification of movement dynamics between groups (opto & control)
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from GridMaze.analysis.core import get_sessions as gs

# %% Global Variables

FRAME_RATE = 60

# %% Functions
