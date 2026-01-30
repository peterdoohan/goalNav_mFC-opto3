"""
Analysis of viral expression blobs in frontal cortex
"""

# %% Imports
import re
import numpy as np
import pandas as pd
from math import ceil, floor
from matplotlib import pyplot as plt

from skimage.measure import label
from scipy.ndimage import gaussian_filter
from brainglobe_atlasapi import BrainGlobeAtlas

from GridMaze.analysis.core.get_anatomy import SubjectAnatomy

# %% Global Variables
from GridMaze.paths import EXPERIMENT_INFO_PATH

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

ATLAS = "allen_mouse_10um"
# %%


def plot_anatomy_summary(
    anatomy_df,
    consider_regions=["PL", "ILA", "ORBm", "ORBvl", "ACAv"],
    colors=["mediumvioletred", "violet", "blueviolet", "darkslateblue", "cornflowerblue"],
    ax=None,
):
    """ """
    df = anatomy_df.groupby(["subject_ID", "simple_name"]).voxels.sum().unstack()
    # set up figure
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(2.5, 3))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.set_ylabel("estimated volume silenced \n (um$^3$)")
    ax.set_xlabel("Subjects")
    # set order of regions
    df = df[consider_regions]
    df.plot(kind="bar", stacked=True, ax=ax, color=colors, alpha=0.5, width=0.8)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=10)
    return


def get_opto_anatomy_df(
    condition="opto",
    restrict_to_fiber_tip=True,
    blob_cleaning_kwargs={"n_blobs": 2, "connectivity": 3},
    cone_mask_kwargs={"cone_angle_deg": 40, "depth_um": 500, "voxel_size_um": 10.0},
    atlas=None,
    atlas_annotations=None,
    verbose=True,
):
    """ """
    # load atlas and annotations
    if atlas is None:
        atlas = BrainGlobeAtlas(ATLAS)
    if atlas_annotations is None:
        atlas_annotations = atlas.annotation

    # loop over subjects
    dfs = []
    subject_IDs = SUBJECT_INFO_DF[SUBJECT_INFO_DF["condition"] == condition].subject_ID.tolist()
    for subject in subject_IDs:
        if verbose:
            print(f"Processing subject {subject}...")
        subject_anatomy = SubjectAnatomy(subject, with_data=["registered_signal", "fiber_coordinates"])
        signal_data = subject_anatomy.registered_signal
        # threshold signal
        thresh_signal = threshold_signal(signal_data, threshold=200)
        # clean thresholded signal (remove nose, keep only large expression zones)
        expression_mask = keep_n_largest_blobs(
            thresh_signal,
            n_blobs=blob_cleaning_kwargs["n_blobs"],
            connectivity=blob_cleaning_kwargs["connectivity"],
        )
        if restrict_to_fiber_tip:
            fiber_coords = subject_anatomy.fiber_coordinates
            masks = []
            for x in ["left", "right"]:
                top_vox = fiber_coords[x]["top"]
                bottom_vox = fiber_coords[x]["bottom"]
                cone_mask = get_cone_mask(
                    fiber_top_voxel=top_vox,
                    fiber_tip_voxel=bottom_vox,
                    volume_shape=signal_data.shape,
                    cone_angle_deg=cone_mask_kwargs["cone_angle_deg"],
                    depth_um=cone_mask_kwargs["depth_um"],
                    voxel_size_um=subject_anatomy.voxel_size_um,
                )
                masks.append(cone_mask)
            fiber_tip_mask = np.logical_or.reduce(masks)
            opto_mask = np.logical_and(expression_mask, fiber_tip_mask)
        else:
            opto_mask = expression_mask

        # get anatomy of corresponding blob (mask)
        df = get_blob_anatomy_summary(opto_mask, atlas, atlas_annotations)
        df["subject_ID"] = subject
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# %% fiber tip masks


def get_cone_mask(
    fiber_top_voxel,
    fiber_tip_voxel,
    volume_shape=(1320, 800, 1140),
    cone_angle_deg=30,
    depth_um=750,
    voxel_size_um=10.0,
):
    fiber_top_voxel = np.asarray(fiber_top_voxel, dtype=float)
    fiber_tip_voxel = np.asarray(fiber_tip_voxel, dtype=float)

    # axis vector in voxel units
    v = fiber_tip_voxel - fiber_top_voxel
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("fiber_top_voxel and fiber_tip_voxel must be different")
    axis = v / norm  # unit vector in voxel coordinates

    nz, ny, nx = volume_shape

    # Convert depth and geometry to voxel units (depth_vox, max radius in voxels)
    depth_vox = float(depth_um) / float(voxel_size_um)
    theta = np.deg2rad(cone_angle_deg)
    max_radius_vox = depth_vox * np.tan(theta)

    # tip position in voxel coordinates (we use tip center as apex)
    tip = fiber_tip_voxel + 0.5  # center of tip voxel, float

    # far endpoint in voxel coordinates (center)
    far = tip + axis * depth_vox

    # compute bounding box (voxel indices) covering tip..far ± max_radius
    min_coords = np.minimum(tip, far) - max_radius_vox
    max_coords = np.maximum(tip, far) + max_radius_vox

    # convert to integer voxel index ranges, clamp to volume bounds
    z0 = max(0, int(floor(min_coords[0])))
    y0 = max(0, int(floor(min_coords[1])))
    x0 = max(0, int(floor(min_coords[2])))

    z1 = min(nz, int(ceil(max_coords[0])) + 1)  # +1 because slice end is exclusive
    y1 = min(ny, int(ceil(max_coords[1])) + 1)
    x1 = min(nx, int(ceil(max_coords[2])) + 1)

    # If bbox is empty (e.g. tip outside volume), return all-False mask
    if (z0 >= z1) or (y0 >= y1) or (x0 >= x1):
        return np.zeros(volume_shape, dtype=bool), (slice(0, 0), slice(0, 0), slice(0, 0))

    # create coordinate arrays using broadcasting (voxel indices, float32)
    # note: use small dtype float32 to save memory/speed
    z_idx = (np.arange(z0, z1, dtype=np.float32) + 0.5).astype(np.float32)  # center coords
    y_idx = (np.arange(y0, y1, dtype=np.float32) + 0.5).astype(np.float32)
    x_idx = (np.arange(x0, x1, dtype=np.float32) + 0.5).astype(np.float32)

    # use broadcasting: shapes (sz,1,1), (1,sy,1), (1,1,sx)
    Z = z_idx[:, None, None]
    Y = y_idx[None, :, None]
    X = x_idx[None, None, :]

    # vectors from tip to each voxel (in voxel units)
    Vz = Z - tip[0]
    Vy = Y - tip[1]
    Vx = X - tip[2]

    # axial distance in voxel units (signed)
    # axis is (dz, dy, dx) in voxel units
    z_axial = (Vz * axis[0]) + (Vy * axis[1]) + (Vx * axis[2])  # shape (sz,sy,sx), float32

    # perpendicular squared distance (avoid extra allocations by computing squared norm)
    # |V|^2 - (z_axial)^2  (because |V_perp|^2 = |V|^2 - (V·u)^2 when u is unit)
    V_sq = (Vz * Vz) + (Vy * Vy) + (Vx * Vx)
    r_sq = V_sq - (z_axial * z_axial)
    # numerical issues: clamp small negative r_sq to zero
    r_sq = np.maximum(r_sq, 0.0)

    # cone test in voxel units: z_axial > 0, z_axial <= depth_vox, r <= z_axial * tan(theta)
    # square both sides to avoid sqrt: r_sq <= (z_axial * tan(theta))**2
    tan_theta = np.tan(theta)
    rhs_sq = (z_axial * tan_theta) ** 2

    # build boolean mask within bounding box
    inside = (z_axial > 0.0) & (z_axial <= depth_vox) & (r_sq <= rhs_sq)

    # allocate full-size output and insert small mask
    cone_mask = np.zeros(volume_shape, dtype=bool)
    cone_mask[z0:z1, y0:y1, x0:x1] = inside

    return cone_mask


def check_fiber_placement(subject, atlas=None, atlas_annotations=None):
    """ """
    if atlas is None:
        atlas = BrainGlobeAtlas(ATLAS, brainglobe_dir=EXPERIMENT_INFO_PATH)
    if atlas_annotations is None:
        atlas_annotations = atlas.annotation

    subject_anatomy = SubjectAnatomy(subject, with_data=["fiber_coordinates"])
    fiber_coords = subject_anatomy.fiber_coordinates
    for f in ["left", "right"]:
        for p in ["top", "bottom"]:
            coord = fiber_coords[f][p]
            voxel = tuple(int(c) for c in coord)
            region_id = atlas_annotations[voxel]
            region_name = atlas.lookup_df[atlas.lookup_df["id"] == region_id]["name"].values[0]
            print(f"Fiber {f} {p} at {voxel} in region: {region_name}")


# %% map from voxels to regions


def get_blob_anatomy_summary(blobs, atlas=None, atlas_annotations=None):
    """
    blobs is boolian numpy array wih voxels in allen mouse brain atlas 10um space
    """
    # load atlas and annotations
    if atlas is None:
        atlas = BrainGlobeAtlas(ATLAS)
    if atlas_annotations is None:
        atlas_annotations = atlas.annotation
    assert blobs.shape == atlas_annotations.shape, "blobs and atlas_annotations must have the same shape"
    # Get unique region IDs in blob
    IDs_in_blob = np.unique(atlas_annotations[blobs])
    IDs_in_blob = IDs_in_blob[IDs_in_blob != 0]

    # Get counts per region ID
    counts = {int(l): int(np.sum((atlas_annotations == l) & blobs)) for l in IDs_in_blob}
    lookup = atlas.lookup_df.set_index("id")  # Map label ids -> names using atlas.lookup_df (id, acronym, name)

    # output datafame
    rows = []
    for lab_id, vox_count in counts.items():
        acronym = lookup.loc[lab_id, "acronym"]
        name = lookup.loc[lab_id, "name"]
        rows.append({"id": lab_id, "acronym": acronym, "name": name, "voxels": vox_count})

    df = pd.DataFrame(rows).sort_values("voxels", ascending=False)
    df["simple_name"] = df.acronym.apply(lambda s: re.match(r"([A-Za-z]+)(.*)", s).groups()[0])
    return df.reset_index(drop=True)


# %% raw signal processing


def get_subject_blobs(subject_ID="mFC-opto_23", thresh_value=400, n_blobs=2, connectivity=3, plot=True):
    """ """
    # load data
    subject_anatomy = SubjectAnatomy(subject_ID, with_data=["registered_signal"])
    # get allen atlas registered viral expression signal
    signal_data = subject_anatomy.registered_signal
    # threshold signal
    thresh_signal = threshold_signal(signal_data, threshold=thresh_value)
    # clean signal
    clean_signal = keep_n_largest_blobs(
        thresh_signal,
        n_blobs=n_blobs,
        connectivity=connectivity,
        verbose=True,
    )
    if plot:
        plot_signal_processing(signal_data, thresh_signal, clean_signal)
    return clean_signal


def threshold_signal(signal, threshold=400):
    """ """
    s = signal.astype(np.float32)
    s_mask = s > threshold
    return s_mask


def keep_n_largest_blobs(thresh_signal, n_blobs=2, connectivity=3, verbose=True):
    """
    Because we have two big injection sites,
    signal can we roughly grouped into two big continous blobs.
    """
    labeled = label(thresh_signal, connectivity=connectivity)

    # count voxels per label
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        raise ValueError("No blobs found in the thresholded signal.")
    counts[0] = 0  # ignore background

    # id labels
    present_labels = np.flatnonzero(counts)
    if n_blobs >= len(present_labels):
        if verbose:
            print("Warning: n_blobs >= number of present blobs, keeping all blobs.")
        # keep all blobs
        return thresh_signal
    # get n largest labels
    top_n_labels = np.argsort(counts[present_labels])[-n_blobs:]
    top_labels = present_labels[top_n_labels]

    # mask for only the n largest blobs
    kept_mask = np.isin(labeled, top_labels)
    return kept_mask


# %% plotting


def plot_signal_processing(
    signal_data,
    thres_signal,
    clean_signal,
    slice_range=np.arange(300, 450, 25),
    cmap="copper",
    axes=None,
):
    """ """
    n_slices = len(slice_range)
    if axes is None:
        fig, axes = plt.subplots(3, n_slices, figsize=(3 * n_slices, 9))
    for i, slice_idx in enumerate(slice_range):
        for j, (data, title) in enumerate(
            zip(
                [signal_data, thres_signal, clean_signal],
                ["Registered Signal", "Thresholded Signal", "Cleaned Signal"],
            )
        ):
            ax = axes[j, i]
            ax.imshow(data[slice_idx, :, :].T, cmap=cmap, origin="lower")
            ax.set_title(f"{title} - {slice_idx}")
            ax.axis("off")
    fig.tight_layout()


def _plot_signal_intensity_histogram(signal_data, n_bins=100, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.spines[["top", "right"]].set_visible(False)
    ax.hist(signal_data.ravel(), bins=n_bins, color="black")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal Intensity")
    ax.set_ylabel("Voxel Count")
