import numpy as np
from astropy.visualization import make_lupton_rgb

# create the empty grid world


def createWorld(height, width, layers, defaultObject):
    world = np.full((height, width, layers), defaultObject)
    return world


# this is a test of making a visual representation


def createWorldImage(world):
    image_r = np.random.random((world.shape[0], world.shape[0]))
    image_g = np.random.random((world.shape[0], world.shape[0]))
    image_b = np.random.random((world.shape[0], world.shape[0]))

    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            image_r[i, j] = world[i, j, 0].appearence[0]
            image_g[i, j] = world[i, j, 0].appearence[1]
            image_b[i, j] = world[i, j, 0].appearence[2]

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    return image
