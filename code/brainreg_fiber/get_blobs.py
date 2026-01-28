"""
code for extracting flourescent plots from serial 2p tomography data in Allen CCFv3 space
@peterdoohan pilfering code from @charlesburns
"""

# %% Imports
import json
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.morphology import remove_small_objects

from brainreg_fiber import run_brainreg as rub

# %% Global Variables
PREPROCESSED_BRAINREG_PATH = Path("../data/preprocessed_data/brainreg")
ATLAS_NAME = "allen_mouse_10um"
VOXEL_SIZE = 10  # in um
ALLEN_ATLAS_INFO_DF = pd.read_csv("./brainreg_fiber/allen_brain_atlas_info.htsv", sep="\t")
AXIS2ATLAS_VECTOR = {
    "ap": [1, 0, 0],
    "si": [0, 1, 0],
    "rl": [0, 0, 1],  # axes in 3D space acccording to 'asr' orientation.
    "pa": [-1, 0, 0],
    "is": [0, -1, 0],
    "lr": [0, 0, -1],
}  # inverses

# %% Functions


def make_signal_df(data, mask):
    coords = np.argwhere(mask)
    values = data[mask]
    df = pd.DataFrame(coords, columns=["i", "j", "k"])
    df["value"] = values
    df["norm_value"] = (values - values.min()) / values.max()  # normalise to [0,1]

    return df


def clean_thresholded_signal(thresholded_signal, min_voxels=1_000, connectivity=3):
    # Step 1: label connected components
    labeled = label(thresholded_signal, connectivity=connectivity)

    # Step 2: remove small components
    # (operates directly on the labeled image)
    cleaned_labeled = remove_small_objects(labeled, min_size=min_voxels)

    # Step 3: convert back to boolean
    cleaned_mask = cleaned_labeled > 0

    return cleaned_mask


def keep_largest_connected_component(
    mask,
    connectivity=3,
):
    """ """
    # Label connected components (0 = background)
    labeled = label(mask, connectivity=connectivity)

    # Fast histogram of label counts: index = label value
    # np.bincount returns length max_label+1; index 0 is background count
    counts = np.bincount(labeled.ravel())

    if counts.size <= 1:
        # only background present (counts has either length 0 or 1)
        return np.zeros_like(mask, dtype=bool)

    # ignore background count (index 0)
    counts[0] = 0

    # label of largest component (this is the label value)
    largest_label = counts.argmax()

    if counts[largest_label] == 0:
        # no foreground voxels
        return np.zeros_like(mask, dtype=bool)

    # build boolean mask for the largest label
    largest_mask = labeled == largest_label
    return largest_mask


def keep_n_largest_components(mask, n=2, connectivity=1, return_labels=False):
    """ """
    if n <= 0:
        kept_mask = np.zeros_like(mask, dtype=bool)
        if return_labels:
            return kept_mask, np.zeros_like(mask, dtype=np.int32)
        return kept_mask

    # Label connected components (0 = background)
    labeled = label(mask, connectivity=connectivity)

    # Fast count of voxels per label (index == label value)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        # only background present
        kept_mask = np.zeros_like(mask, dtype=bool)
        if return_labels:
            return kept_mask, np.zeros_like(mask, dtype=labeled.dtype)
        return kept_mask

    # Zero out background count
    counts[0] = 0

    # Identify labels that are present (non-zero count)
    present_labels = np.flatnonzero(counts)
    if present_labels.size == 0:
        kept_mask = np.zeros_like(mask, dtype=bool)
        if return_labels:
            return kept_mask, np.zeros_like(mask, dtype=labeled.dtype)
        return kept_mask

    # If n >= number of present components, keep them all
    if n >= present_labels.size:
        kept_mask = labeled > 0
        if return_labels:
            return kept_mask, labeled
        return kept_mask

    # Otherwise pick the labels of the top-n largest components
    # argsort on counts of present_labels (ascending) then take last n
    top_n_idx = np.argsort(counts[present_labels])[-n:]
    top_labels = present_labels[top_n_idx]

    # Create boolean mask of voxels belonging to any of the top labels
    kept_mask = np.isin(labeled, top_labels)

    if return_labels:
        # produce a label volume containing only the top labels (others zero)
        preserved_labels = np.zeros_like(labeled)
        # keep original label numbers so downstream regionprops remain valid
        preserved_labels[kept_mask] = labeled[kept_mask]
        return kept_mask, preserved_labels

    return kept_mask


def threshold_signal(signal_data, gamma=1.11):
    """ """
    norm = signal_data.astype(np.float32)
    norm -= norm.min()
    norm /= norm.max()
    corr = norm**gamma
    thresh = threshold_otsu(corr)
    thresholded_signal = corr > thresh
    return thresholded_signal


def thresh_gaussian_otsu(signal_data, sigma=1.0, gamma=1.0):
    img = signal_data.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img /= img.max()
    # optional gamma
    img = img**gamma
    # smooth to reduce noise
    sm = gaussian_filter(img, sigma=sigma)
    t = threshold_otsu(sm)
    return sm > t


def thresh_multi_otsu(signal_data, n_classes=3, keep_classes=(1, 2)):
    img = signal_data.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img /= img.max()
    thresholds = threshold_multiotsu(img, classes=n_classes)
    # digitize into class labels 0..n_classes-1
    labels = np.digitize(img, bins=thresholds)
    # keep specified classes as foreground
    mask = np.isin(labels, keep_classes)
    return mask


def basic_threshold(signal_data, thresh_value=3_000):
    img = signal_data.astype(np.float32)
    mask = img > thresh_value
    return mask


def get_data(brainreg_path, signal_channel=2):
    """
    load various .tiff files from brainreg output folder,
    returning as a dictionary of numpy arrays.
        keys: 'signal_data', 'atlas_registration_data', 'boundaries',
        'deformation_field_0', 'deformation_field_1', 'deformation_field_2'
    """
    signal_channel = str(signal_channel)
    labels_and_filenames = [
        ("signal_data", f"downsampled_{signal_channel}.tiff"),
        ("atlas_registration_data", "registered_atlas.tiff"),
        ("boundaries", "boundaries.tiff"),
    ] + [(f"deformation_field_{i}", f"deformation_field_{i}.tiff") for i in range(3)]
    data = {}
    for label, filename in labels_and_filenames:
        try:
            data.update({label: tifffile.imread(brainreg_path / ATLAS_NAME / filename)})
        except Exception as e:
            print(f"Could not load {filename}: \n {e}")
    return data
