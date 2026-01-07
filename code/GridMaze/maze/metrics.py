"""
Lib for quantifying properties of mazes, between-ness centriality, mean shortest-path distance, etc.
@ peterdoohan
"""

# %% Imports
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

from GridMaze.maze import representations as mr

# %% Global variables

# %% Functions


def get_betweeness_centrality(simple_maze, with_edges=False):
    """ """
    if with_edges:
        graph = mr.get_extended_simple_maze(simple_maze)
    else:
        graph = simple_maze.copy()

    BC = nx.betweenness_centrality(graph)
    coord2label = mr.get_maze_coord2label(simple_maze)
    return {coord2label[coord]: bc for coord, bc in BC.items()}


def get_mean_geodesic_distance(simple_maze, with_edges=False):
    """ """
    if with_edges:
        graph = mr.get_extended_simple_maze(simple_maze)
    else:
        graph = simple_maze.copy()

    SPD = dict(nx.all_pairs_shortest_path_length(graph))
    coord2label = mr.get_maze_coord2label(simple_maze)
    label2spd = {}
    for coord1, spd_dict in SPD.items():
        label1 = coord2label[coord1]
        spd_list = []
        for coord2, spd in spd_dict.items():
            label2 = coord2label[coord2]
            if label1 != label2:
                spd_list.append(spd)
        label2spd[label1] = np.mean(spd_list)
    return label2spd


def get_fitness(simple_maze, with_edges=False):
    """
    calculates the mean euclidean vs geodesic distance correlation for each node in the maze
    (corr vector of euclidean distances to all other nodes vs geodesic distances to all other nodes)
    """
    if with_edges:
        graph = mr.get_extended_simple_maze(simple_maze)
    else:
        graph = simple_maze.copy()

    # get shortest path and euclidean distances between all pairs of nodes
    SPD = dict(nx.all_pairs_shortest_path_length(graph))
    coord2label = mr.get_maze_coord2label(simple_maze)
    label2position = mr.get_maze_label2position(simple_maze)
    label2decorr = {}
    for cord, d_dict in SPD.items():
        label = coord2label[cord]
        pos = label2position[label]
        eds, gds = [], []
        for cord2, gd in d_dict.items():
            label2 = coord2label[cord2]
            pos2 = label2position[label2]
            eds.append(euclidean(pos, pos2))
            gds.append(gd)
        label2decorr[label] = 1 - np.corrcoef(eds, gds)[0, 1]

    return label2decorr


def get_node_degree(simple_maze, with_edges=False):
    """ """
    if with_edges:
        graph = mr.get_extended_simple_maze(simple_maze)
    else:
        graph = simple_maze.copy()

    ND = dict(graph.degree())
    coord2label = mr.get_maze_coord2label(simple_maze)

    return {coord2label[coord]: nd for coord, nd in ND.items()}
