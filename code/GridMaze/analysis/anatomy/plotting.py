"""
plot anatomy summaries of viral expression and fiber placement
"""

# %% Imports
import vedo
from brainrender.actors import Volume
from brainrender import Scene
import brainglobe_heatmap as bgh
from matplotlib import pyplot as plt

# %% Global Variables

# %% Functions


def make_scene(blob):
    """ """
    scene = Scene(atlas_name="allen_mouse_10um")
    actor = Volume(blob, as_surface=True, c="Reds")
    scene.add(actor)


def test_plot(visualise_regions=["PL"]):
    # initialise brainrender scene
    scene = Scene(atlas_name="allen_mouse_10um", title="", root=True)
    scene.plotter.axes = False
    for region in visualise_regions:
        scene.add_brain_region(region, alpha=0.15, hemisphere="left", color="magenta", silhouette=True)
    scene.render()
    return scene
