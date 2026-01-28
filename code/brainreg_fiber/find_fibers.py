"""
Map manually traced fibers in raw data (sample space) to Allen CCFv3 space
@peterdoohan with code pilfered from @charlesburns
"""

# %% Imports
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from nibabel.orientations import axcodes2ornt, ornt_transform
import tifffile
import matplotlib.pyplot as plt
import brainglobe_heatmap as bgh


# %% Global Variables
RAW_HISTOLOGY_PATH = Path("../data/raw_data/histology")  # Contains /<subject_ID>/brainsaw_output
PREPROCESSED_BRAINREG_PATH = Path(
    "../data/preprocessed_data/brainreg"
)  # Will contain brainreg/<subject_ID>/<atlas_name>/outputs

ATLAS_NAME = "allen_mouse_10um"
SUBJECT_IDS = [p.stem for p in RAW_HISTOLOGY_PATH.iterdir() if p.name != "raw_compressed"]
SIGNAL_CHANNEL = 2  # red flourophore

# %% Functions


def find_fiber_top_and_bottom(points):
    """ """
    # Fit line via SVD (total least squares)
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    direction = Vt[0]  # unit vector

    # Identify top and bottom points by y-coordinate
    top_point = points[np.argmin(points[:, 1])]
    bottom_point = points[np.argmax(points[:, 1])]

    # Orthogonal projections onto the line
    t_top = np.dot(top_point - centroid, direction)
    t_bottom = np.dot(bottom_point - centroid, direction)

    top_proj = centroid + t_top * direction
    bottom_proj = centroid + t_bottom * direction

    return top_proj, bottom_proj


def load_data(subject_dir):
    """ """
    # load defomation fields for mapping from sample to atlas space
    deform_fields = {}
    for i in range(3):
        deform_fields[i] = tifffile.imread(subject_dir.brainreg_path / ATLAS_NAME / f"deformation_field_{i}.tiff")

    # load manually labelled fiber tracks in sample space coords (left and right)
    img_shape = get_img_shape(subject_dir.input_path)
    voxel_sizes = get_voxel_sizes(subject_dir.recipe_path)
    fiber_tracts = {}
    for label in ["left", "right"]:
        df = pd.read_csv(subject_dir.histology_path / f"fiber_{label}.csv")
        df.drop(columns="index", inplace=True)
        df.columns = ["x", "y", "z"]
        sample_points = transform_points_to_atlas_grid(
            df,
            img_shape=img_shape,
            original_voxel_sizes_um=(voxel_sizes["Z"], voxel_sizes["Y"], voxel_sizes["X"]),
            atlas_voxel_sizes_um=(10, 10, 10),
            point_columns=("x", "y", "z"),  # XML points are in z,y,x order
            from_orientation="psl",  # histology orientation in i,j,k, i.e. z,y,x order
            to_orientation="asr",  # Standard atlas orientation
        )
        atlas_points = sample_coords_to_allen_space(
            sample_points[["i", "j", "k"]].to_numpy(),
            deform_fields,
        )
        fiber_tracts[label] = atlas_points

    return fiber_tracts


def sample_coords_to_allen_space(
    points,
    deform_fields,
):
    """
    Not super trivial transformation from sample coordinates to allen coords
    """
    atlas_mm = []
    for point in points:
        atlas_mm.append(
            [
                deform_fields[0][int(point[0]), int(point[1]), int(point[2])],
                deform_fields[1][int(point[0]), int(point[1]), int(point[2])],
                deform_fields[2][int(point[0]), int(point[1]), int(point[2])],
            ]
        )
    atlas_um = np.array(atlas_mm) * 1000
    return atlas_um


# %% load manually labelled points


def test(subject_dir, fiber="left"):
    """ """
    voxel_sizes = get_voxel_sizes(subject_dir.recipe_path)
    img_shape = get_img_shape(subject_dir.input_path)
    df = pd.read_csv(subject_dir.histology_path / f"fiber_{fiber}.csv")
    df.drop(columns="index", inplace=True)
    df.columns = ["x", "y", "z"]
    atlas_points = transform_points_to_atlas_grid(
        df,
        img_shape=img_shape,
        original_voxel_sizes_um=(voxel_sizes["Z"], voxel_sizes["Y"], voxel_sizes["X"]),
        atlas_voxel_sizes_um=(10, 10, 10),
        point_columns=("x", "y", "z"),  # XML points are in z,y,x order
        from_orientation="psl",  # histology orientation in i,j,k, i.e. z,y,x order
        to_orientation="asr",  # Standard atlas orientation
    )
    return atlas_points


def get_voxel_sizes(recipe_path):
    """
    Extract voxel sizes from brainreg recipe YAML file.

    Parameters
    ----------
    recipe_path : str or Path
        Path to the recipe YAML file

    Returns
    -------
    dict
        Voxel sizes with keys ['X', 'Y', 'Z'] in micrometers
    """
    recipe_path = Path(recipe_path)
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(recipe_path, "r") as file:
        try:
            params = yaml.safe_load(file)
            voxel_sizes = params["VoxelSize"]
            print(f"Loaded voxel sizes from {recipe_path}: {voxel_sizes}")
            return voxel_sizes
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(f"Could not parse voxel sizes from {recipe_path}: {e}")


def get_img_shape(brainreg_input_path: Path):
    """
    Determine the shape of the original image stack.

    Parameters
    ----------
    brainreg_input_path : Path
        Directory containing the image files

    Returns
    -------
    tuple
        Image shape as (n_slices, height, width)

    Notes
    -----
    Assumes all images have the same height and width, and counts
    the number of files to determine the number of slices.
    """
    input_path = Path(brainreg_input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Count image files (assuming they're all the same size)
    image_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    if not image_files:
        raise ValueError(f"No TIFF files found in {input_path}")

    n_slices = len(image_files)

    # Get dimensions from first image
    first_image = tifffile.imread(image_files[0])
    height, width = first_image.shape[:2]  # Handle both 2D and 3D arrays

    shape = (n_slices, height, width)

    return shape


def transform_points_to_atlas_grid(
    points_df,
    img_shape,
    original_voxel_sizes_um,
    atlas_voxel_sizes_um=(10.0, 10.0, 10.0),
    point_columns=("z", "y", "x"),
    from_orientation="asr",
    to_orientation="asr",
):
    """
    Transform points from original image space to atlas voxel space.

    This function handles:
    1. Axis reorientation (e.g., posterior→anterior flip)
    2. Voxel size scaling
    3. Coordinate system transformation

    Parameters
    ----------
    points_df : pd.DataFrame
        Points with columns specified by point_columns
    img_shape : tuple
        Shape of original image as (dim0, dim1, dim2)
    original_voxel_sizes_um : tuple
        Voxel sizes in micrometers as (size0, size1, size2)
    atlas_voxel_sizes_um : tuple
        Target atlas voxel sizes in micrometers
    point_columns : tuple
        Column names in points_df corresponding to the three dimensions
    from_orientation : str
        Original orientation code (e.g., 'psl' = Posterior, Superior, Left)
    to_orientation : str
        Target orientation code (e.g., 'asr' = Anterior, Superior, Right)

    Returns
    -------
    pd.DataFrame
        Transformed points with columns ['i', 'j', 'k'] in atlas space

    Notes
    -----
    Orientation codes follow brainglobe convention:
    - First letter: anterior (a) or posterior (p)
    - Second letter: superior (s) or inferior (i)
    - Third letter: left (l) or right (r)

    Common orientations:
    - 'asr': Anterior, Superior, Right (standard atlas)
    - 'psl': Posterior, Superior, Left (common histology)
    """
    print(f"Transforming coordinates from '{from_orientation}' to '{to_orientation}' orientation")

    # Extract point coordinates
    coords = points_df[list(point_columns)].to_numpy().astype(float)

    # Step 1: Reorient the coordinate axes
    # This is where the anterior-posterior flip happens!
    coords_reoriented, orientation_transform = reorient_coordinates(coords, img_shape, from_orientation, to_orientation)

    # Step 2: Convert voxel indices to physical coordinates (mm)
    original_voxel_sizes_mm = np.array(original_voxel_sizes_um) / 1000.0
    atlas_voxel_sizes_mm = np.array(atlas_voxel_sizes_um) / 1000.0

    # Reorder voxel sizes to match the new orientation
    reoriented_voxel_sizes_mm = reorder_voxel_sizes(original_voxel_sizes_mm, orientation_transform)

    # Convert to physical coordinates, then to atlas voxel indices
    physical_coords_mm = coords_reoriented * reoriented_voxel_sizes_mm
    atlas_voxel_coords = physical_coords_mm / atlas_voxel_sizes_mm

    # Create output dataframe
    result = points_df.copy()
    result[["i", "j", "k"]] = atlas_voxel_coords

    return result


def reorient_coordinates(coords, img_shape, from_orientation, to_orientation):
    """
    Reorient coordinate points between different axis orientations.

    This function handles the core axis transformations, including:
    - Axis permutations (e.g., swapping x and z)
    - Axis inversions (e.g., posterior → anterior flip)

    Parameters
    ----------
    coords : np.ndarray
        Input coordinates with shape (n_points, 3)
    img_shape : tuple
        Original image dimensions
    from_orientation : str
        Source orientation (e.g., 'psl')
    to_orientation : str
        Target orientation (e.g., 'asr')

    Returns
    -------
    reoriented_coords : np.ndarray
        Coordinates in the new orientation
    transform_matrix : np.ndarray
        The transformation matrix applied
    """
    # Convert brainglobe codes to nibabel format
    from_axes = orientation_code_to_nibabel(from_orientation)
    to_axes = orientation_code_to_nibabel(to_orientation)

    # Get transformation matrix
    from_ornt = axcodes2ornt(from_axes)
    to_ornt = axcodes2ornt(to_axes)
    transform_matrix = ornt_transform(from_ornt, to_ornt)

    # Apply transformation
    coords = np.asarray(coords, dtype=float)
    reoriented_coords = np.empty_like(coords)

    for output_axis in range(3):
        input_axis = int(transform_matrix[output_axis, 0])
        direction = int(transform_matrix[output_axis, 1])  # +1 normal, -1 flipped

        if direction == 1:
            # Normal direction: copy values directly
            reoriented_coords[:, output_axis] = coords[:, input_axis]
        else:
            # Flipped direction: invert relative to image boundary
            # This is where anterior-posterior flipping occurs!
            max_coord = img_shape[input_axis] - 1
            reoriented_coords[:, output_axis] = max_coord - coords[:, input_axis]

    return reoriented_coords, transform_matrix


def reorder_voxel_sizes(voxel_sizes_mm, transform_matrix):
    """
    Reorder voxel sizes to match reoriented coordinate axes.

    When we reorient coordinates, we also need to reorder the voxel sizes
    to match the new axis arrangement.
    """
    reordered_sizes = np.empty(3, dtype=float)

    for output_axis in range(3):
        input_axis = int(transform_matrix[output_axis, 0])
        reordered_sizes[output_axis] = voxel_sizes_mm[input_axis]

    return reordered_sizes


def orientation_code_to_nibabel(orientation_code):
    """
    Convert brainglobe orientation code to nibabel axis codes.

    Parameters
    ----------
    orientation_code : str
        3-letter orientation code (e.g., 'asr', 'psl')

    Returns
    -------
    tuple
        Corresponding nibabel axis codes
    """
    code_map = {
        "a": "A",  # anterior
        "p": "P",  # posterior
        "s": "S",  # superior
        "i": "I",  # inferior
        "l": "L",  # left
        "r": "R",  # right
    }

    if len(orientation_code) != 3:
        raise ValueError(f"Orientation code must be 3 letters, got: {orientation_code}")

    return tuple(code_map[char.lower()] for char in orientation_code)


# %% test plotting


def test_plot(
    points,
):
    values = dict(PL=0, ILA=0, DP=0, ACAd=1, ACAv=1)  # scalar values for each region
    # build custom CMAP:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 1.3])
    scene1 = bgh.Heatmap(
        values,
        position=(points[0][:, 0].mean(), 0, 0),
        orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        thickness=10,
        title="",
        format="2D",
    ).plot_subplot(fig, axes[0])

    scene2 = bgh.Heatmap(
        values,
        position=(0, 0, points[0][:, 2].mean()),
        orientation="sagittal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        thickness=10,
        title="",
        format="2D",
    ).plot_subplot(fig, axes[1])

    for ax in axes:
        ax.axis("off")

    colorbar1 = fig.axes[-1]
    colorbar1.remove()
    colorbar2 = fig.axes[-1]
    colorbar2.remove()
    for pts in points:
        axes[0].scatter(pts[:, 2], pts[:, 1], color="red", s=1)
        axes[1].scatter(pts[:, 0], pts[:, 1], color="red", s=1)
    fig.tight_layout()
