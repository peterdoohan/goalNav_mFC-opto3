"""This file registers video pixel coordinates of select towers on a maze for later use to correct for fish-eye distortion. A quality control step is included
to ensure that the camera does not move over the recording period."""

# %% imports
import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.spatial.distance import euclidean
from GridMaze.maze.representations import _get_node_positions_dict, get_simple_nodes_dict

# %% Global variables
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


from GridMaze.paths import EXPERIMENT_INFO_PATH, VIDEO_PATH

ALIGNMENT_POINTS = ["A1", "A4", "A7", "C3", "C5", "D1", "D4", "D7", "E3", "E5", "G1", "G4", "G7"]

SUBJECT_INFO_DF = pd.read_csv(EXPERIMENT_INFO_PATH / "subject_info_df.htsv", sep="\t")

with open(EXPERIMENT_INFO_PATH / "maze_info.json") as input_file:
    MAZE_INFO = json.load(input_file)


# %% Main Function
def get_maze_registration_df(
    big_maze_rig, session_date=None, maze_registration_path=EXPERIMENT_INFO_PATH / "maze_registration.htsv"
):
    """Load maze registration file if it exists, otherwise run registration. Returns dataframe with the pixel and physical coordinates
    of each maze tower used for later alignmnet steps."""
    try:
        maze_registration_df = pd.read_csv(str(maze_registration_path), sep="\t", header=[0, 1])
        maze_registration_df.columns = pd.MultiIndex.from_tuples(
            [(a, b) if not "Unnamed" in b else (a, "") for a, b in maze_registration_df.columns]
        )
        maze_registration_df[("date", "")] = pd.to_datetime(maze_registration_df[("date", "")]).dt.date
    except FileNotFoundError:
        print(f"No maze registration file found in {EXPERIMENT_INFO_PATH}. Running registration now.")
        maze_registration_df = run_maze_registration()
    # filter for big maze rig
    maze_registration_df = maze_registration_df[maze_registration_df.big_maze_rig == big_maze_rig]
    if session_date is not None:
        if isinstance(session_date, str):
            session_date = date.fromisoformat(session_date)
        _dates = maze_registration_df.date.unique()
        # get most recent date before or on the given date
        valid_dates = [d for d in _dates if d <= session_date]
        if len(valid_dates) == 0:
            raise ValueError(f"No registration found for big maze rig {big_maze_rig} on or before date {date}")
        closest_date = max(valid_dates)
        maze_registration_df = maze_registration_df[maze_registration_df.date == closest_date]
    return maze_registration_df.reset_index(drop=True)


def run_maze_registration(n_click_replicates=1, save=True):
    """ """
    #
    alignment_point2simple_node = get_simple_nodes_dict()
    simple_node2physical_coords = _get_node_positions_dict()
    #
    reg_dfs = []
    for big_maze_rig in ["L", "R"]:
        print(f"Running maze registration for big maze rig {big_maze_rig}")
        sample_video_paths = get_sample_videos(big_maze_rig)
        date2registration_df = {}
        registration_df = None
        for video_path in sample_video_paths:
            _date = datetime.strptime(video_path.name.split("_")[-1].split(".")[0], "%Y-%m-%d-%H%M%S").date()
            if registration_df is not None:
                use_previous = use_previous_registration(video_path, registration_df)
                if use_previous:
                    continue
                else:
                    registration_df = get_registration_df(video_path, n_click_replicates=n_click_replicates)
                    date2registration_df[_date.isoformat()] = registration_df
            else:
                registration_df = get_registration_df(video_path, n_click_replicates=n_click_replicates)
                date2registration_df[_date.isoformat()] = registration_df
        for date, registration_df in date2registration_df.items():
            _df = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("alignment_point", ""),
                        ("pixel", "x"),
                        ("pixel", "y"),
                        ("physical", "x"),
                        ("physical", "y"),
                        ("date", ""),
                        ("big_maze_rig", ""),
                    ]
                )
            )
            _df[("alignment_point", "")] = registration_df.columns
            _df[("pixel", "x")] = registration_df.loc["mean_x"].values
            _df[("pixel", "y")] = registration_df.loc["mean_y"].values
            physical_coords = [
                simple_node2physical_coords[alignment_point2simple_node[ap]] for ap in registration_df.columns
            ]
            _df[("physical", "x")] = [pc[0] for pc in physical_coords]
            _df[("physical", "y")] = [pc[1] for pc in physical_coords]
            _df[("date", "")] = date
            _df[("big_maze_rig", "")] = big_maze_rig
            reg_dfs.append(_df)
    df = pd.concat(reg_dfs, axis=0).reset_index(drop=True)
    if save:
        df.to_csv(EXPERIMENT_INFO_PATH / "maze_registration.htsv", sep="\t", index=False)
    return df


def get_registration_df(video_path, n_click_replicates=3):
    tower2pixel_coord_replicates = []
    for _ in range(n_click_replicates):
        tower2pixel_coord_replicates.append(get_alignment_point_pixel_coords(video_path, ALIGNMENT_POINTS))
    registration_df = pd.DataFrame(tower2pixel_coord_replicates).apply(calculate_replicate_mean_and_std)
    return registration_df


def use_previous_registration(video_path, registration_df, scaling_factor=5, color="red"):
    """ """
    # set up fig
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    # get image to display on plot
    video = cv2.VideoCapture(str(video_path))
    temp_image_path = VIDEO_PATH / "temp_image.png"
    ret, first_frame = video.read()
    cv2.imwrite(str(temp_image_path), first_frame)
    image = cv2.imread(str(temp_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orientation = [0, image.shape[1], 0, image.shape[0]]
    ax.imshow(image, extent=orientation)
    # plot previous registration
    for tower in registration_df.columns:
        mean_x = registration_df.loc["mean_x", tower]
        mean_y = registration_df.loc["mean_y", tower]
        std_x = registration_df.loc["std_x", tower]
        std_y = registration_df.loc["std_y", tower]
        radius = max(std_x, std_y)
        radius = 1 if radius == 0 else radius
        radius = radius * scaling_factor
        circle = plt.Circle((mean_x, mean_y), radius, color=color, alpha=0.5)
        ax.add_patch(circle)
    # ask if user wants to keep registration
    plt.title("Do you want to keep this registration for the current video?")

    plt.show(block=True)  # Display the figure

    # Now prompt the user in the console
    while True:
        user_input = input("Keep this registration? (y/n): ").lower()
        if user_input == "y":
            plt.close(f)  # Close the figure
            return True
        elif user_input == "n":
            plt.close(f)  # Close the figure
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def get_sample_videos(big_maze_rig):
    """
    Returns example video paths recorded on a given big maze rig for each exp date.
    Args:
    - big_maze_rig: str, 'L' or 'R', the big maze rig to sample videos from.
    """
    rig_subjects = SUBJECT_INFO_DF[SUBJECT_INFO_DF["big_maze_rig"] == big_maze_rig].subject_ID.values
    # assign an example subject to search for videos
    for sub in rig_subjects:
        example_subject = sub
        video_paths = [p for p in VIDEO_PATH.iterdir() if example_subject in p.name and p.suffix == ".mp4"]
        if len(video_paths) > 0:
            break
    if len(video_paths) == 0:
        raise ValueError(f"No subjects with video data found for big maze rig {big_maze_rig}")
    video_dates = [
        datetime.strptime(p.name.split("_")[-1].split(".")[0], "%Y-%m-%d-%H%M%S").date() for p in video_paths
    ]
    # remove duplicate dates and associated paths
    unique_dates = []
    unique_video_paths = []
    for date, p in zip(video_dates, video_paths):
        if date not in unique_dates:
            unique_dates.append(date)
            unique_video_paths.append(p)
    # order by date
    ordered_video_paths = [p for _, p in sorted(zip(unique_dates, unique_video_paths))]
    return ordered_video_paths


# %% Supporting Functions


def get_alignment_point_pixel_coords(video_path, alignment_points):
    video = cv2.VideoCapture(str(video_path))
    temp_image_path = VIDEO_PATH / "temp_image.png"
    ret, first_frame = video.read()
    cv2.imwrite(str(temp_image_path), first_frame)
    image = cv2.imread(str(temp_image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alignment_point2pixel_coord = {}
    for point in alignment_points:
        print(f"Click on {point}")
        pixel_coord = get_pixel_coords_from_image(image, label=point)[0]
        alignment_point2pixel_coord[point] = pixel_coord
    # remove temp image
    temp_image_path.unlink()
    return alignment_point2pixel_coord


def calculate_replicate_mean_and_std(column):
    x_values = [item[0] for item in column]
    y_values = [item[1] for item in column]
    return pd.Series(
        {"mean_x": np.mean(x_values), "mean_y": np.mean(y_values), "std_x": np.std(x_values), "std_y": np.std(y_values)}
    )


def plot_click_registration(registration_click_dfs, scaling_factor=5):
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    pallet = sns.color_palette("tab10", n_colors=len(registration_click_dfs))
    for reg_df, color in zip(registration_click_dfs, pallet):
        for tower in reg_df.columns:
            mean_x = reg_df.loc["mean_x", tower]
            mean_y = reg_df.loc["mean_y", tower]
            std_x = reg_df.loc["std_x", tower]
            std_y = reg_df.loc["std_y", tower]
            radius = max(std_x, std_y) * scaling_factor
            radius = 1 if radius == 0 else radius
            circle = plt.Circle((mean_x, mean_y), radius, color=color, alpha=0.5)
            ax.add_patch(circle)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(0, 1200)  # Set appropriate limits for x and y axis based on your data
    ax.set_ylim(0, 1000)
    return


def get_cross_session_variance(registration_click_dfs):
    """Returns the standard deviation of the distance between the pixel coordinates of the same tower in a list of tower_coord_dicts"""
    n = len(registration_click_dfs)
    towers = registration_click_dfs[0].columns
    permutations = [(i, j) for i in range(n) for j in range(n) if i < j]
    offset_distances = []
    for a, b in permutations:
        for t in towers:
            pos_a = (registration_click_dfs[a][t]["mean_x"], registration_click_dfs[a][t]["mean_y"])
            pos_b = (registration_click_dfs[b][t]["mean_x"], registration_click_dfs[b][t]["mean_y"])
            offset_distance = euclidean(pos_a, pos_b)
            offset_distances.append(offset_distance)
    return np.std(offset_distances)


def get_average_alignment_coords(registration_click_dfs):
    combined_click_dfs = pd.concat(registration_click_dfs)
    tower_x_means = combined_click_dfs.loc["mean_x"].mean()
    tower_y_means = combined_click_dfs.loc["mean_y"].mean()
    average_tower_coords = {tower: (tower_x_means[tower], tower_y_means[tower]) for tower in tower_x_means.keys()}
    return average_tower_coords


def get_pixel_coords_from_image(image, label):
    """Returns pixel coordinates of a click on an image"""
    plt.figure(figsize=(15, 10))
    orientation = [0, image.shape[1], 0, image.shape[0]]
    plt.imshow(image, extent=orientation)
    plt.title(f"click on the center of {label}")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    pixel_coords = plt.ginput(n=1, timeout=0, show_clicks=True)
    plt.close()
    return pixel_coords


# %% Other


def get_image_size_from_video():
    """Returns the image size of the videos in the raw_data directory, using 1st frame from the 1st video as an example image
    Output is (height, width)"""
    example_video_path = str(get_sample_videos("L")[0])
    temp_image_path = os.path.join(VIDEO_PATH, "temp_image.png")
    video = cv2.VideoCapture(example_video_path)
    temp_image_path = os.path.join(VIDEO_PATH, "temp_image.png")
    ret, first_frame = video.read()
    if ret:
        cv2.imwrite(temp_image_path, first_frame)
    else:
        print(f"Could not read first frame from {example_video_path}")
    image = cv2.imread(temp_image_path)
    image_size = image.shape[:2]
    os.remove(temp_image_path)
    return image_size
