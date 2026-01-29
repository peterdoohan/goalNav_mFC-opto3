"""
Save subject level anaomty data to processed_data
"""

# %% Imports
import json
import shutil
import pandas as pd

from GridMaze.preprocessing import get_data_directory as dd

# %% Global Variables
from GridMaze.paths import PROCESSED_DATA_PATH, EXPERIMENT_INFO_PATH

with open(EXPERIMENT_INFO_PATH / "condition2virus.json", "r") as infile:
    CONDITION2VIRUS = json.load(infile)

ALLEN_ATALAS_RES = 10  # um

SIGNAL_CHANNEL = 2

SIGNAL_COLOR = "red"

# %% Functions


def process_anatomy_data(
    data_structures=["registered_signal", "fiber_coordinates", "anatomy_info"],
    overwrite=False,
    verbose=True,
):
    """ """
    subject_data_directory = dd.get_subject_data_directory()
    for subject_dir in subject_data_directory.itertuples():
        if verbose:
            print(f"Processing anatomy data for subject {subject_dir.subject_ID}...")
        if "registered_signal" in data_structures:
            if verbose:
                print("  - saving registered virus signal...")
            save_registered_virus_signal(subject_dir, overwrite=overwrite)
        if "fiber_coordinates" in data_structures:
            if verbose:
                print("  - saving fiber coordinates...")
            save_fiber_coordinates(subject_dir, overwrite=overwrite)
        if "anatomy_info" in data_structures:
            if verbose:
                print("  - saving anatomy info...")
            save_anatomy_info(subject_dir, overwrite=overwrite)
    if verbose:
        print("Anatomy data processing complete.")


def save_registered_virus_signal(subject_dir, overwrite=False):
    """ """
    # define new data path
    subject_data_path = PROCESSED_DATA_PATH / subject_dir.subject_ID / "anatomy"
    subject_data_path.mkdir(parents=True, exist_ok=True)
    new_data_path = subject_data_path / "registered_signal.tiff"
    # flouroescent signal warped to allen atlas space in preprocessing
    brainreg_output = subject_dir.brainreg_path / "allen_mouse_10um" / f"downsampled_standard_{SIGNAL_CHANNEL}.tiff"
    # execute copy
    if not new_data_path.exists() or overwrite:
        shutil.copyfile(brainreg_output, new_data_path)


def save_fiber_coordinates(subject_dir, overwrite=False):
    """ """
    # define new data path
    subject_data_path = PROCESSED_DATA_PATH / subject_dir.subject_ID / "anatomy"
    subject_data_path.mkdir(parents=True, exist_ok=True)
    new_data_path = subject_data_path / "fiber_coordinates.json"
    # load manually labelled fiber top and bottom coords from brainreg (labelled in atlas space)
    fiber_coordinates = {}
    for label in ["left", "right"]:
        _coords = {}
        for position in ["top", "bottom"]:
            manual_label_path = subject_dir.brainreg_path / "fiber_coordinates" / f"{label}_{position}.csv"
            coord_tuple = tuple(pd.read_csv(manual_label_path, header=None).iloc[0])  # in allen atlas voxels
            _coords[position] = coord_tuple
        fiber_coordinates[label] = _coords
    # save to json
    if not new_data_path.exists() or overwrite:
        with open(new_data_path, "w") as outfile:
            json.dump(fiber_coordinates, outfile, indent=4)


def save_anatomy_info(subject_dir, overwrite=False):
    """ """
    info = {
        "virus": CONDITION2VIRUS[subject_dir.condition],
        "atlas": "allen_mouse_10um",
        "signal_channel": SIGNAL_CHANNEL,
        "signal_color": SIGNAL_COLOR,
        "n_fibers": 2,
        "n_injection_sites": 2,
        "region_targeted": "PL",
        "hemisphere": "both",
    }
    # save info json
    subject_data_path = PROCESSED_DATA_PATH / subject_dir.subject_ID / "anatomy"
    subject_data_path.mkdir(parents=True, exist_ok=True)
    new_data_path = subject_data_path / "anatomy_info.json"

    if not new_data_path.exists() or overwrite:
        with open(new_data_path, "w") as outfile:
            json.dump(info, outfile, indent=4)
