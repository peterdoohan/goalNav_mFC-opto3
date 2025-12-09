"""This module contains functions for correcting for radial distortion in the video images,
and for converting raw pixel coordinates to physical coordinates."""

# %% Imports
import numpy as np
import pylab as plt
from datetime import date
from scipy.optimize import minimize
from GridMaze.preprocessing import maze_registration as maze_reg
from scipy.spatial.distance import euclidean

# %% Initial global variables
IMAGE_SIZE = maze_reg.get_image_size_from_video()


# %% Functions for non-linear distortion correction (raw_pixels -> distorted/corrected_pixels)
def distort_pixel_coords(pixel_coords: tuple, strength: int) -> tuple:
    """Apply radial distortion to the set of (x,y) pixel coordinates, using the radial distortion function.
    strength>0 gives barrel distortion, strength<0 gives pincushion distortion, strength=0 gives no distortion."""
    centered_pixel_coords = center_pixel_coords(pixel_coords)  # center coordinates
    x, y = vectorise_coordinates(centered_pixel_coords)
    y_size, x_size = IMAGE_SIZE
    x_distort, y_distort = radial_distortion(x, y, x_size, y_size, strength)  # tuplise coordinates
    distorted_centered_coords = [(i, j) for i, j in zip(x_distort, y_distort)]
    return uncenter_pixel_coords(distorted_centered_coords)  # uncenter coordinates


def plot_radial_distortion(pixel_coords, strength):
    """Plot the effect of radial distortion on a set of (x,y) coordinates (showing both original coords and distorted coords),
    using the function distort_pixel_coords"""
    x, y = vectorise_coordinates(pixel_coords)
    distorted_pixel_coords = distort_pixel_coords(pixel_coords, strength)
    x_distort, y_distort = vectorise_coordinates(distorted_pixel_coords)
    plt.figure(1, clear=True, figsize=(5, 5))
    plt.scatter(x, y, color="k", label="Original")
    plt.scatter(x_distort, y_distort, color="r", label="Distorted")
    plt.legend()


def radial_distortion(x, y, x_size, y_size, strength):
    """Apply radial distortion to the set of x,y points. strength>0
    gives barrel distortion, strength<0 gives pincushion distortion,
    strength=0 gives no distortion."""
    r = strength * np.sqrt(x**2 + y**2) / np.sqrt(x_size**2 + y_size * 2)
    r[r == 0] = 1e-6  # Avoid divide by 0
    if strength > 0:
        theta = np.arctan(r) / r
    else:
        theta = r / np.arctan(r)
    xn = x * theta
    yn = y * theta
    return xn, yn


def vectorise_coordinates(coordinates: tuple):
    """Vectorise a set of (x,y) coordinates"""
    x = np.array([i[0] for i in coordinates])
    y = np.array([i[1] for i in coordinates])
    return x, y


def center_pixel_coords(pixel_coords):
    y_size, x_size = IMAGE_SIZE
    return [center_coordinate(x, y, x_size, y_size) for x, y in pixel_coords]


def center_coordinate(x, y, x_size, y_size):
    """Centers a coordinate to the center of an image (where the original origin was in the bottom left corner)"""
    x_center = x_size / 2
    y_center = y_size / 2
    x_new = x - x_center
    y_new = y - y_center
    return x_new, y_new


def uncenter_pixel_coords(centered_pixel_coords):
    y_size, x_size = IMAGE_SIZE
    return [uncenter_coordinate(x, y, x_size, y_size) for x, y in centered_pixel_coords]


def uncenter_coordinate(x, y, x_size, y_size):
    """Uncenters a set of (x,y) coordinates to new origin at bottom left corner"""
    x_center = x_size / 2
    y_center = y_size / 2
    x_new = x + x_center
    y_new = y + y_center
    return x_new, y_new


# %% Find optimal distortion strength for correcting non-linear video distortion


def get_optimal_distortion_strength(big_maze_rig, session_date, initial_guess=0.0):
    """Calculates the optimal distortion strength for correcting non-linear video distortion.
    i.e finds the distortion strength that, when applied, removes radial distortion from the video,
    and raw pixel coords."""
    registration_df = maze_reg.get_maze_registration_df(big_maze_rig, session_date=session_date)
    physical_coords = [tuple(x) for x in registration_df.physical.values]
    pixel_coords = [tuple(x) for x in registration_df.pixel.values]
    physical_coords_distance_vector = coord_distance_vector(physical_coords)
    result = minimize(
        distortion_cost,
        initial_guess,
        args=(physical_coords_distance_vector, pixel_coords),
        method="BFGS",
        options={"eps": 1e-4},
    )  # smaller step sizes don't find minima
    return result.x[0]


def distortion_cost(strength, physical_coords_distance_vector, pixel_coords):
    """Calculates the cost of a distortion strength, by comparing the relative distances between
    all points in both distorted_pixel and physical space"""
    distorted_pixel_coords = distort_pixel_coords(pixel_coords, strength)
    distorted_pixel_coords_distance_vector = coord_distance_vector(distorted_pixel_coords)
    return np.sum((physical_coords_distance_vector - distorted_pixel_coords_distance_vector) ** 2)


def coord_distance_vector(coords, normalised=True):
    """Calculates the distance between all points in a set of (x,y) coordinates"""
    distances = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distances.append(euclidean(coords[i], coords[j]))
    if normalised:
        return np.array(distances) / np.std(distances)
    return np.array(distances)


# %% Functions for linear transformtation between corrected_pixel and physical coordinates
def get_linear_transormation_matrix(big_maze_rig, session_date):
    """Finds the Linear Transformation Matrix (T) that maps the distortion-corrected pixel coordinates to the physical coordinates.
    From calibration coordinates in CALIBRATION_COORDINATES_DF,"""
    registration_df = maze_reg.get_maze_registration_df(big_maze_rig, session_date=session_date)
    physical_coords = [tuple(x) for x in registration_df.physical.values]
    pixel_coords = [tuple(x) for x in registration_df.pixel.values]
    dist_strength = get_optimal_distortion_strength(big_maze_rig, session_date)
    corrected_pixel_coords = distort_pixel_coords(pixel_coords, strength=dist_strength)
    physical_system = np.asarray(physical_coords)
    corrected_pixel_system = np.asarray(corrected_pixel_coords)
    # solve the least squares problem X * A = Y
    X = homogenise(corrected_pixel_system)
    Y = homogenise(physical_system)
    T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return T


def homogenise(x):
    """Homogenise a matrix x by appending a column of ones"""
    return np.hstack([x, np.ones((x.shape[0], 1))])


# %% Full raw pixel to physical coordinate translation functions
def translate_pixel2physical_coords(raw_pixel_coords, big_maze_rig, session_date):
    """Translates a set of raw pixel coordinates to physical coordinates.
    Input can be a tuple or list of tuples."""
    optimal_distortion_strength = get_optimal_distortion_strength(big_maze_rig, session_date)
    lt_matrix = get_linear_transormation_matrix(big_maze_rig, session_date)
    if isinstance(raw_pixel_coords, tuple):
        raw_pixel_coords = [raw_pixel_coords]
    corrected_pixel_coords = distort_pixel_coords(raw_pixel_coords, strength=optimal_distortion_strength)
    physical_coords = translate_corrected_pixel2physical_coords(corrected_pixel_coords, lt_matrix)
    if len(physical_coords) == 1:
        return physical_coords[0]
    return physical_coords


def translate_corrected_pixel2physical_coords(corrected_pixel_coords, lt_matrix):
    """Applies the linear transformation matrix (VIDEO_LT_MATRIX) to a set of distortion corrected pixel coordinates."""
    unhomog = lambda x: x[:, :-1]
    transform = lambda x: unhomog(np.dot(homogenise(x), lt_matrix))
    return transform(np.asarray(corrected_pixel_coords))


# %% Quality control functions
def pixel_to_position_tests(big_maze_rig="L", session_date=date(2025, 11, 14)):
    """Checks that distortion correction looks sensible and prints the optimal distortion strength value found.
    Also checks the accuracy of complete coordinate transformation is acceptable. If either of these checks
    an assertion error is raised."""
    registration_df = maze_reg.get_maze_registration_df(big_maze_rig, session_date=session_date)
    optimal_distortion_strength = get_optimal_distortion_strength(big_maze_rig, session_date=session_date)
    lt_matrix = get_linear_transormation_matrix(big_maze_rig, session_date=session_date)
    cal_physical_coords = [tuple(x) for x in registration_df.physical.values]
    cal_pixel_coords = [tuple(x) for x in registration_df.pixel.values]
    estimated_physical_coords = translate_pixel2physical_coords(
        cal_pixel_coords, big_maze_rig=big_maze_rig, session_date=session_date
    )
    print(f"optimal distortion strength: {optimal_distortion_strength}")
    plot_radial_distortion(cal_pixel_coords, strength=optimal_distortion_strength)
    plt.title("Distortion Correction Visualisation")
    plt.show()
    print(f"video linear transormation matrix: {lt_matrix}")
    error = np.abs(cal_physical_coords - estimated_physical_coords)
    print(f"mean translation error: {np.mean(error):.3f} m")
    print(f"std translation error: {np.std(error):.3f} m")
    plt.figure(1, clear=True, figsize=(5, 5))
    plt.scatter(*vectorise_coordinates(cal_physical_coords), color="k", label="True")
    plt.scatter(*vectorise_coordinates(estimated_physical_coords), color="r", label="Estimated")
    plt.legend()
    plt.show()
    assert optimal_distortion_strength != 0, "Finding Optimal Distortion Strength Failed"
    assert np.allclose(
        cal_physical_coords, estimated_physical_coords, atol=0.01
    ), "Pixel to Physical Coordinate Translation Failed"
