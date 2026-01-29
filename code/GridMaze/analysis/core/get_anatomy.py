"""
Simple object for loading subject anatomy data from disk for analyses
@peterdoohan
"""

# %% Imports
import json
import pandas as pd
import tifffile

# %% Global Variables

# %% Functions

from GridMaze.paths import EXPERIMENT_INFO_PATH, PROCESSED_DATA_PATH

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

ANATOMY_DATA_STRUCTURE2FILENAME = {
    "registered_signal": "registered_signal.tiff",
    "fiber_coordinates": "fiber_coordinates.json",
    "anatomy_info": "anatomy_info.json",
}


# %% Subject Anatomy Object


class SubjectAnatomy:
    """ """

    def __init__(self, subject_ID, with_data="all"):
        self.has_data = with_data
        self.subject_ID = subject_ID
        self.condition = SUBJECT_INFO_DF.loc[SUBJECT_INFO_DF["subject_ID"] == subject_ID, "condition"].values[0]
        processed_data_path = PROCESSED_DATA_PATH / subject_ID / "anatomy"
        # load anatomy info
        with open(processed_data_path / "anatomy_info.json", "r") as infile:
            anatomy_info = json.load(infile)
        for attr_name, attr_value in anatomy_info.items():
            setattr(self, attr_name, attr_value)
        # load data structures
        for attr_name, filename in ANATOMY_DATA_STRUCTURE2FILENAME.items():
            if with_data == "all" or attr_name in with_data:
                file_path = processed_data_path / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"Anatomy data structure {attr_name} not found for subject {subject_ID}")
                if attr_name == "registered_signal":
                    setattr(self, attr_name, tifffile.imread(file_path))
                elif attr_name in ["fiber_coordinates", "anatomy_info"]:
                    with open(file_path, "r") as infile:
                        data = json.load(infile)
                    setattr(self, attr_name, data)
                else:
                    raise ValueError(f"Anatomy data structure {attr_name} not recognised")
        return

    def __repr__(self):
        """Return a nicely formatted string representation of the object."""
        return (
            f"-SubjectAnatomy-----------------------------------------------------\n"
            f"    Subject ID: {self.subject_ID}, Condition: {self.condition}\n"
            f"    Virus: {self.virus}, Target Region: {self.region_targeted}\n"
        )
