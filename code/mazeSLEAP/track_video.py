"""
This script contains functions to track body-part positions from GridMaze experiments using SLEAP, running parallel jobs on
a SLURM managed HPC."

Before running script, ensure that SLEAP models have been moved from the local computer where they were developed to ./models
Further, ensure that ./jobs/slurm, ./jobs/out, and ./jobs/err folders exist in the working directory.

@peterdoohan
"""

# %% imports
import sleap
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import os
import json

# %% Global variables

# this folder should contain one centroid & centered instance model for each experimental session type
SLEAP_MODELS_PATH = Path("mazeSLEAP/models")

MODEL_TYPE2NAME = {
    "untethered": "big_maze_opto_untethered",
    "tethered": "big_maze_opto_tethered",
    "retrain_tethered": "20-22_fix",
}

VIDEO_PATH = Path("../data/raw_data/video")
SLEAP_PATH = Path("../data/preprocessed_data/SLEAP")

# retrieve date from which subjects were fiber-tethered (different SLEAP models for tethered/untethered)
EXPERIMENT_INFO_PATH = Path("../data/experiment_info/")
with open(EXPERIMENT_INFO_PATH / "fiber_tethered_date.json", "r") as f:
    TETHERED_DATE = date.fromisoformat(json.load(f))

SUBJECT_INFO_DF = pd.read_csv("../data/experiment_info/subject_info_df.htsv", sep="\t")


# %% # rerun tracking for back tracking sessions eith retrained model (only subjects on R maze)
R_MAZE_SUBJECTS = SUBJECT_INFO_DF[SUBJECT_INFO_DF.big_maze_rig == "R"].subject_ID.to_list()
RERUN_DATES = [
    date(2025, 11, 20),
    date(2025, 11, 21),
    date(2025, 11, 22),
]

# %% Functions


def run_sleap_preprocessing():
    """ """
    video_paths_df = get_video_paths_df()
    video_paths_df = video_paths_df[~video_paths_df.tracking_completed]
    if video_paths_df.empty:
        print("All videos have been tracked. No new videos to track.")
        return
    # check jobs folders exist
    for jobs_folder in ["slurm", "out", "err"]:
        if not Path(f"mazeSLEAP/jobs/{jobs_folder}").exists():
            os.mkdir(f"mazeSLEAP/jobs/{jobs_folder}")
    for session_info in video_paths_df.itertuples():
        print(f"Submitting {session_info.video_path} to HPC")
        script_path = get_sleap_SLURM_script(session_info)
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")
    print("All video tracking jobs submitted to HPC. Check progress with 'squeue -u <username>'")


def get_sleap_SLURM_script(video_info, RAM="32GB", time_limit="24:00:00"):
    """
    Writes a SLURM script to run sleap tracking on the video from a session specified in video_info.
    Input: video_info: pd.Series, with columns: subject_ID, session_type, datetime, video_path (row from the output of get_video_paths_df())
    Output: script_path: str, path to the SLURM script (saved in mazeSLEAP/jobs/slurm/)
    """
    tethered = "tethered" if video_info.tethered else "untethered"
    session_ID = f"{video_info.subject_ID}_{tethered}_{video_info.datetime.isoformat()}"
    # hard code dates that need to get rerun with retained model
    if (video_info.datetime.date() in RERUN_DATES) & (video_info.subject_ID in R_MAZE_SUBJECTS):
        tethered = "retrain_tethered"
    # write SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name=sleap_tracking_{session_ID}
#SBATCH --output=mazeSLEAP/jobs/out/sleap_tracking_{session_ID}.out
#SBATCH --error=mazeSLEAP/jobs/err/sleap_tracking_{session_ID}.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --exclude=gpu-sr670-21
#SBATCH --exclude=gpu-sr670-22
#SBATCH --exclude=gpu-380-10
#SBATCH --mem={RAM}
#SBATCH --time={time_limit}

echo $SLURMD_NODENAME
module load miniconda
module load SLEAP
nvidia-smi
conda deactivate
conda activate sleap
hostname

python -c "from mazeSLEAP import track_video; track_video.track_video('{video_info.video_path}', '{tethered}')"
"""
    script_path = f"mazeSLEAP/jobs/slurm/sleap_tracking_{session_ID}.sh"
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def track_video(video_path, sleap_model, save_labels=True, return_labels=False):
    """Uses SLEAP API to load a raw_data video, load a top-down SLEAP inference model, and predict the labels for the video."""
    # load video & inference model
    video = sleap.load_video(str(video_path), grayscale=True)
    print(f"tracking video {video_path}")
    sleap_predictor = load_sleap_predictor(sleap_model)
    # process video
    predictions = sleap_predictor.predict(video)
    # save results
    output_filename = ".".join(Path(video_path).name.split(".")[:-1]) + f".predicted_{datetime.now().isoformat()}.h5"
    if save_labels:
        predictions.export(str(SLEAP_PATH / output_filename))
    if return_labels:
        return predictions


def load_sleap_predictor(sleap_model="untethered", batch_size=16):
    """
    Searches the models folder to find the correct models for a given session type and returns a sleap predictor object.

    Args
    - session_type: str, the type of session in your experiment (eg, maze, open_field etc.)
    """
    all_model_paths = [p for p in SLEAP_MODELS_PATH.iterdir() if p.is_dir()]
    model_types = ["centroid", "centered_instance"]
    model_paths = []
    for model_type in model_types:
        model_path = [
            p
            for p in all_model_paths
            if p.name.split(".")[0] == MODEL_TYPE2NAME[sleap_model] and p.name.split(".")[1] == model_type
        ]
        if len(model_path) != 1:
            raise FileNotFoundError(
                f"Check there is only one centroid and one centered instance model for session type {sleap_model} in {SLEAP_MODELS_PATH}"
            )
        else:
            model_path = model_paths.append(str(model_path[0]))
    sleap_predictor = sleap.load_model(
        model_paths,
        batch_size=batch_size,
        tracker_max_instances=1,
        max_instances=1,
        disable_gpu_preallocation=True,
        progress_reporting="json",
    )
    return sleap_predictor


def get_video_paths_df():
    """
    Returns a pd.Dataframe with data extracted from video filenames,
        rows: sessions,
        columns:
            subject_ID: str
            session_type: str (maze, open_field, etc..)
            datetime: datetime object
            video_path: str (relative path to raw video file)
            tracking_completed: bool (if sleap tracking has been performed on video from session)
    """
    all_video_files = [f.name for f in Path(VIDEO_PATH).iterdir() if f.suffix == ".mp4"]
    all_sleap_files = [f.name for f in Path(SLEAP_PATH).iterdir() if f.suffix == ".h5"]
    all_sleap_original_datetime_strings = [  # datetimes_strings of original videos predictions were made on
        f.split(".")[0].split("_", 2)[-1] for f in all_sleap_files
    ]
    video_paths_info = []
    for video_file in all_video_files:
        subject_ID = "_".join(video_file.split("_")[:2])
        datetime_string = video_file.split("_")[-1].split(".")[0]
        _datetime = datetime.strptime(datetime_string, "%Y-%m-%d-%H%M%S")
        tethered = True if _datetime.date() >= TETHERED_DATE else False

        tracking_completed = True if datetime_string in all_sleap_original_datetime_strings else False
        video_paths_info.append(
            {
                "subject_ID": subject_ID,
                "datetime": _datetime,
                "tethered": tethered,
                "video_path": str(Path(VIDEO_PATH) / video_file),
                "tracking_completed": tracking_completed,
            }
        )
    video_paths_df = pd.DataFrame(video_paths_info)
    return video_paths_df


# %% Main
if __name__ == "__main__":
    # check necessary folders exits
    if not SLEAP_MODELS_PATH.exists():
        raise FileNotFoundError(f"Models folder not found at {SLEAP_MODELS_PATH}")
    elif not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video folder not found at {VIDEO_PATH}")
    elif not SLEAP_PATH.exists():
        raise FileNotFoundError(f"SLEAP folder not found at {SLEAP_PATH}")

    run_sleap_preprocessing()
