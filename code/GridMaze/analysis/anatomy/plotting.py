"""
plot anatomy summaries of viral expression and fiber placement
"""

# %% Imports
import vedo
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from brainrender import Scene
from brainrender.actors import Volume

from GridMaze.analysis.anatomy import blobs as ab
from GridMaze.analysis.core.get_anatomy import SubjectAnatomy

# %% Global Variables
vedo.settings.default_backend = "vtk"

from GridMaze.paths import EXPERIMENT_INFO_PATH

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    SUBJECT_IDS = json.load(infile)

# %% Core plotting


def plot_anatomy_summary(condition="opto"):
    """"""
    avg_signal, avg_fiber_coords = get_subject_average_data(condition=condition)
    scene = plot_brainrender_anatomy(
        signal=avg_signal,
        fiber_coordinates=avg_fiber_coords,
        voxel_size=10,
        downsample_to=100,
        signal_thresh=250 if condition == "opto" else 5000,
        visualise_regions=["PL", "ILA", "ORBm"],
        region_colors=[
            "mediumvioletred",
            "violet",
            "blueviolet",
        ],
        extend_fiber=100,
        whole_brain=True,
    )
    return


def plot_brainrender_anatomy(
    signal=None,
    fiber_coordinates=None,
    voxel_size=10,
    downsample_to=100,
    signal_thresh=250,  # control use 5000
    visualise_regions=["PL", "ILA", "ORBm"],
    region_colors=[
        "mediumvioletred",
        "violet",
        "blueviolet",
    ],
    extend_fiber=100,
    whole_brain=True,
):
    # initialise brainrender scene
    scene = Scene(atlas_name="allen_mouse_10um", title="", root=whole_brain)
    scene.plotter.background((1, 1, 1))
    scene.plotter.axes = False

    # plot optic fibers as cylinders
    if fiber_coordinates is not None:
        for fiber in ["left", "right"]:
            top_coord = np.array(fiber_coordinates[fiber]["top"]) * voxel_size
            bottom_coord = np.array(fiber_coordinates[fiber]["bottom"]) * voxel_size
            new_top = extend_point_along_line(top_coord, bottom_coord, extend_fiber * voxel_size)
            # transpose x and z
            new_top_coord = [new_top[2], new_top[1], new_top[0]]
            bottom_coord = [bottom_coord[2], bottom_coord[1], bottom_coord[0]]
            cyl = vedo.shapes.Cylinder(
                pos=[new_top_coord, bottom_coord],
                c="powderblue",
                r=100,
                alpha=0.9,
            )
            scene.add(cyl)

    # plot virus as blob
    if signal is not None:
        # clip noise
        _signal = np.copy(signal)
        thres_mask = ab.threshold_signal(_signal, signal_thresh)
        clean_mask = ab.keep_n_largest_blobs(thres_mask, n_blobs=1, connectivity=3)
        _signal[~clean_mask] = 0
        # downsample volume
        factor = downsample_to // voxel_size
        ds_signal = downsample_mean(_signal, factor)
        # homog
        ds_signal[ds_signal > 0] = 100
        # add blob to scene
        vol = Volume(
            ds_signal.astype(np.float64),
            voxel_size=downsample_to,
            min_value=0.5,
            cmap="Reds_r",
        )
        vol.alpha(0.3)
        scene.add(vol)

    # add brain regions
    for region, color in zip(visualise_regions, region_colors):
        scene.add_brain_region(region, alpha=0.1, color=color, silhouette=True)

    scene.render()
    return scene


def extend_point_along_line(top, bottom, n_um):
    """
    Extend the line from bottom -> top by n_um beyond top.
    """
    direction = top - bottom
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("top and bottom points are identical")

    unit_dir = direction / norm
    new_top = top + unit_dir * n_um

    return new_top


def downsample_mean(volume, factor):
    """
    Downsample a 3D volume by an integer factor using block averaging.
    """
    z, y, x = volume.shape

    # crop so dimensions are divisible by factor
    zc = (z // factor) * factor
    yc = (y // factor) * factor
    xc = (x // factor) * factor

    volume = volume[:zc, :yc, :xc]

    # reshape into blocks and average
    volume_ds = volume.reshape(zc // factor, factor, yc // factor, factor, xc // factor, factor).mean(axis=(1, 3, 5))

    return volume_ds


# %% Get summary blobs for plotting


def get_subject_average_data(condition="opto", verbose=True, smooth_SD=1):
    """
    Get average registered signal and fiber coordinates for subjects in a given condition.
    """
    fiber_coords = {"left": {"top": [], "bottom": []}, "right": {"top": [], "bottom": []}}
    subject_IDs = SUBJECT_INFO_DF[SUBJECT_INFO_DF["condition"] == condition].subject_ID.tolist()
    signals = np.zeros((len(subject_IDs), 1320, 800, 1140), dtype=np.float32)
    for i, subject in enumerate(subject_IDs):
        if verbose:
            print(f"Processing subject {subject}...")
        subject_anatomy = SubjectAnatomy(subject, with_data=["registered_signal", "fiber_coordinates"])
        # get average registered signal
        if smooth_SD:
            signals[i] = gaussian_filter(subject_anatomy.registered_signal.astype(np.float32), sigma=smooth_SD)
        else:
            signals[i] = subject_anatomy.registered_signal.astype(np.float32)
        # get fiber coordinates
        for fiber in ["left", "right"]:
            for t in ["top", "bottom"]:
                coord = subject_anatomy.fiber_coordinates[fiber][t]
                fiber_coords[fiber][t].append(coord)
    if verbose:
        print("Averaging data across subjects...")
    # average registered signal
    avg_signal = signals.mean(axis=0)
    # average fiber coordinates
    avg_fiber_coords = {
        "left": {
            "top": np.mean(fiber_coords["left"]["top"], axis=0).tolist(),
            "bottom": np.mean(fiber_coords["left"]["bottom"], axis=0).tolist(),
        },
        "right": {
            "top": np.mean(fiber_coords["right"]["top"], axis=0).tolist(),
            "bottom": np.mean(fiber_coords["right"]["bottom"], axis=0).tolist(),
        },
    }
    return avg_signal, avg_fiber_coords


# %%
