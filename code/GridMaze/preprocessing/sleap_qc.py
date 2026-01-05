"""
Quick script to manually QC sleap outputs to see if any sessions need to be rerun.
"""

# %% Imports
from datetime import date
from matplotlib import pyplot as plt
from GridMaze.preprocessing import get_data_directory as dd
from GridMaze.preprocessing import get_frames_dfs as fd
from GridMaze.maze import representations as mr
from GridMaze.maze import plotting as mp

# %% Global variables
from GridMaze.paths import RESULTS_PATH

RESULTS_DIR = RESULTS_PATH / "tracking_qc"

DATA_DIRECTORY = dd.get_sessions_data_directory()

# %%


def save_session_trajectory_qc_plots(subject_ID=None, date=None, bodypart="head_back", save=True, overwrite=False):
    """ """
    if subject_ID is not None:
        _data_dir = DATA_DIRECTORY[DATA_DIRECTORY.subject_ID == subject_ID]
    if date is not None:
        _data_dir = _data_dir[_data_dir.date == date]
    else:
        _data_dir = DATA_DIRECTORY

    for session_dir in _data_dir.itertuples():
        if not isinstance(session_dir.SLEAP_path, str):
            continue
        save_path = RESULTS_DIR / f"{session_dir.subject_ID}" / f"{session_dir.date.isoformat()}.png"
        if save and (save_path.exists() and not overwrite):
            print(
                f"{session_dir.subject_ID}.{session_dir.date.isoformat()} - already exists. Use overwrite=True to save new qc plot."
            )
            continue
        tracking_df = fd.get_tracking_df(session_dir)
        x, y = tracking_df.x.values, tracking_df.y.values
        simple_maze = mr.get_simple_maze(session_dir.maze_name)
        plot_session_trajectory(
            simple_maze, x, y, session_dir.subject_ID, session_dir.date, save=save, overwrite=overwrite
        )


def plot_session_trajectory(simple_maze, x, y, subject_ID, date, ax=None, save=True, overwrite=False):
    save_path = RESULTS_DIR / f"{subject_ID}" / f"{date.isoformat()}.png"
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    # plot maze outline
    mp.plot_simple_maze_silhouette(simple_maze, ax=ax, color="silver")
    # plot trajectory
    ax.plot(x, y, color="black", alpha=0.5, linewidth=0.5)
    f.suptitle(f"{subject_ID}.{date.isoformat()}")
    if save:
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        f.savefig(save_path)
