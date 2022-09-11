from astropy.visualization import make_lupton_rgb
import numpy as np


def agent_visualfield(world, location, k=4, wall_app=[153.0, 51.0, 102.0]):
    """
    Create an agent visual field of size (2k + 1, 2k + 1) pixels
    """

    bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)
    # instantiate image
    image_r = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
    image_g = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
    image_b = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))

    for i in range(bounds[0], bounds[1] + 1):
        for j in range(bounds[2], bounds[3] + 1):
            # while outside the world array index...
            if i < 0 or j < 0 or i >= world.shape[0] - 1 or j >= world.shape[1]:
                # image has shape bounds[1] - bounds[0], bounds[3] - bounds[2]
                # visual appearance = wall
                image_r[i - bounds[0], j - bounds[2]] = wall_app[0]
                image_g[i - bounds[0], j - bounds[2]] = wall_app[1]
                image_b[i - bounds[0], j - bounds[2]] = wall_app[2]
            else:
                image_r[i - bounds[0], j - bounds[2]] = world[i, j, 0].appearence[0]
                image_g[i - bounds[0], j - bounds[2]] = world[i, j, 0].appearence[1]
                image_b[i - bounds[0], j - bounds[2]] = world[i, j, 0].appearence[2]

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    return image
