import numpy as np
from astropy.visualization import make_lupton_rgb


def create_world(height, width, layers, defaultObject):
    """
    Create the empty grid world
    """

    world = np.full((height, width, layers), defaultObject)
    return world


def create_world_image(world, layers=0):
    """
    Output an RGB of god view
    TODO: test to make sure that the shapes are correct for height and width
    """

    image_r = np.random.random((world.shape[0], world.shape[1]))
    image_g = np.random.random((world.shape[0], world.shape[1]))
    image_b = np.random.random((world.shape[0], world.shape[1]))

    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            image_r[i, j] = world[i, j, layers].appearence[0]
            image_g[i, j] = world[i, j, layers].appearence[1]
            image_b[i, j] = world[i, j, layers].appearence[2]

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    return image
