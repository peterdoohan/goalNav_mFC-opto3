"""
Library for loading processed and analysis data from disk so it is appropriately stored in memory
@peterdoohan
"""

# %% Imports
import json
import numpy as np
import pandas as pd

# %% Global Variables


# %% Functions


def load(filepath):
    """Loads processed/analysis data from disk into memory. Making adjustments to the saved data as necessary."""
    # check filepath exists
    if not filepath.exists():
        return None
    data_set = filepath.parts[-4]  # processed or analysis data
    if data_set == "processed_data":
        return _processed_data(filepath)
    elif data_set == "analysis_data":
        return _analysis_data(filepath)
    else:
        raise ValueError(f"Data set {data_set} not recognised, must  in folder processed_data or analysis_data")


def _processed_data(filepath):
    """Loads processed data from disk into memory. Making adjustments to the saved data as necessary."""
    processed_data_structure = filepath.name
    if processed_data_structure == "session_info.json":
        with open(filepath, "r") as infile:
            return json.load(infile)
    elif processed_data_structure == "trials.htsv":
        df = pd.read_csv(filepath, sep="\t")
        df = _unflatten_df_columns(df)
        df[("stim_trial", "")] = [eval(i) if i == "False" else i for i in df.stim_trial]
        return df
    elif processed_data_structure == "events.htsv":
        df = pd.read_csv(filepath, sep="\t")
        return df
    elif processed_data_structure == "frames.tracking.htsv":
        df = pd.read_csv(filepath, sep="\t")
        df = _unflatten_df_columns(df)
        return df
    elif processed_data_structure == "frames.trajectories.htsv":
        df = pd.read_csv(filepath, sep="\t")
        df = _unflatten_df_columns(df)
        return df
    elif processed_data_structure == "frames.trialInfo.htsv":
        df = pd.read_csv(filepath, sep="\t")
        return df
    else:
        raise ValueError(f"Processed data structure {processed_data_structure} not recognised")


def _analysis_data(filepath):
    """Loads analysis data from disk into memory. Making adjustments to the saved data as necessary."""
    analysis_data_structure = filepath.name
    if analysis_data_structure == "frames.navigation.parquet":
        return _load_multiindex_parquet(filepath)
    elif analysis_data_structure == "navigation_strategies_dataframe.parquet":
        df = _load_multiindex_parquet(filepath)
        # fix columns with multiple data types
        df[("stim_on", "")] = df[("stim_on", "")].apply(_rebool)
        for d in ["N", "S", "E", "W"]:
            df[("available", d)] = df[("available", d)].map(_rebool)
        return df
    elif analysis_data_structure == "trajectory_decisions_dataframe.parquet":
        return _load_multiindex_parquet(filepath)
    else:
        raise ValueError(f"Analysis data structure {analysis_data_structure} not recognised")


def _rebool(x):
    if x == 0:
        return False
    elif x == 1:
        return True
    else:
        return x


def _unflatten_df_columns(df):
    """
    Unflattens columns when loading a multi-index columns from disk, see
    populate_processed_data.get_flattered_multiindex_columns to see how the columns were flattened
    and saved.
    """
    df.columns = pd.MultiIndex.from_tuples([tuple(col.split(".")) if "." in col else (col, "") for col in df.columns])
    return df


def _load_multiindex_parquet(filepath):
    df = pd.read_parquet(filepath)
    if not np.all([isinstance(i, tuple) for i in df.columns]):
        if all([len(i.split(",")) > 1 for i in df.columns]):  # deal with multiindex
            df.columns = pd.MultiIndex.from_tuples([eval(col) for col in df.columns])
        else:
            pass
    df[df.isna()] = np.nan  # convert None values to np.nan
    return df
