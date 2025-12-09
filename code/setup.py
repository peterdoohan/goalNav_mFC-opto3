"""
convience commands for analysis & data processing
"""

# %% track video
# !cd code
from mazeSLEAP import track_video as tv

df = tv.get_video_paths_df()
df[~df.tracking_completed]
# tv.run_sleap_preprocessing()

# %% get dev session

from GridMaze.analysis.core import get_sessions as gs

session = gs.get_maze_sessions(subject_IDs=["mFC-opto_23"], maze_names=["maze_1"], days_on_maze=[21])[0]
