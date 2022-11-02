from astropy.visualization import make_lupton_rgb
import numpy as np


def agent_visualfield(world, location, tile_size, k=4,
                      out_of_bounds_colour=[50.0, 50.0, 50.0]):
    """Compute the visual field of the agent.

    Args:
      world: The world in which the agent is located.
      location: The location of the agent. A tuple of (x, y) or (x, y, layer)
        coordinates.
      tile_size: The size of the tiles in the world. A tuple of (w, h) dimensions.
      k: The vision radius of the agent.
      out_of_bounds_colour: The color to use for out of bounds tiles. A tuple of
        (r, g, b) values.

    Returns:
      A `((2k + 1) * tile_size[0], (2k + 1) * tile_size[1], 3)` numpy array
      describing the visual field of the agent centered at the given location,
      with vision radius `k`.
    """
    # Ensure location is (x, y) or (x, y, layer) coordinates.
    assert len(location) == 2 or len(location) == 3,\
        "location must be a tuple of (x, y) or (x, y, layer) coordinates"
    # Ensure tile_size is (w, h) dimensions.
    assert len(tile_size) == 2,\
        "tile_size must be a tuple of (w, h) dimensions"

    out_of_bounds_appearance = np.fill((tile_size[0], tile_size[1], 3),
                                       out_of_bounds_colour)

    x, y = location[:2]
    layer = location[2] if len(location) == 3 else 0
    tile_w, tile_h = tile_size

    visual_field = np.zeros((tile_w * (2 * k + 1), tile_h * (2 * k + 1), 3))
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            # Compute global (non-relative) coordinates of the tile
            px, py = x + i, y + j
            # Check if the pixel is outside the world
            if px < 0 or py < 0 or px >= world.shape[0] or py >= world.shape[1]:
              tile = out_of_bounds_appearance
            else:
              # Compute the visual field tile for the current location
              tile = world[px, py, layer].visual_field_tile(tile_size)

            # Compute the location of the visual field tile
            tile_x = (i + k) * tile_w
            tile_y = (j + k) * tile_h
            # Add the visual field tile to the visual field.
            visual_field[tile_x:tile_x + tile_w, tile_y:tile_y + tile_h] = tile

    return visual_field
