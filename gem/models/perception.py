import numpy as np


def agent_visualfield(world, location, tile_size, k=4,
                      out_of_bounds_colour=[50.0, 50.0, 50.0],
                      convert_to_float=True):
    """Compute the visual field of the agent.

    Args:
      world: The world in which the agent is located.
      location: The location of the agent. A tuple of (x, y) or (x, y, layer)
        coordinates.
      tile_size: The size of the tiles in the world. A tuple of (w, h) dimensions.
      k: The vision radius of the agent. Use None for infinite vision (i.e. the
        agent can see the entire world).
      out_of_bounds_colour: The color to use for out of bounds tiles. A tuple of
        (r, g, b) values.
      convert_to_float: Whether to convert the visual field to a float array
        between 0 and 1. If False, the visual field will be an integer array
        between 0 and 255.

    Returns:
      A `((2k + 1) * tile_size[0], (2k + 1) * tile_size[1], 3)` numpy array
      describing the visual field of the agent centered at the given location,
      with vision radius `k`.
    """
    # Ensure location is (x, y) or (x, y, layer) coordinates.
    assert len(location) == 2 or len(location) == 3,\
        'location must be a tuple of (x, y) or (x, y, layer) coordinates'
    # Ensure tile_size is (w, h) dimensions.
    assert len(tile_size) == 2,\
        'tile_size must be a tuple of (w, h) dimensions'
    # Ensure k is a positive integer or None.
    assert k is None or (isinstance(k, int) and k >= 0),\
        'k must be a positive integer or None (for infinite vision)'

    out_of_bounds_appearance = np.full((tile_size[0], tile_size[1], 3),
                                       out_of_bounds_colour) / 255.0

    x, y = location[:2]
    layer = location[2] if len(location) == 3 else 0
    tile_w, tile_h = tile_size

    if k is None:
      # Infinite vision
      x_min, x_max = 0, world.shape[0]
      y_min, y_max = 0, world.shape[1]
    else:
      # Finite vision
      x_min, x_max = -k, k + 1
      y_min, y_max = -k, k + 1

    width, height = (x_max - x_min) * tile_w, (y_max - y_min) * tile_h
    visual_field = np.zeros((width, height, 3))

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            # Compute global (non-relative) coordinates of the tile
            px, py = x + i, y + j
            # Check if the pixel is outside the world
            if px < 0 or py < 0 or px >= world.shape[0] or py >= world.shape[1]:
              tile = out_of_bounds_appearance
            else:
              tile = world[px, py, layer].appearance

            # Compute the location of the visual field tile
            x0, x1 = (i - x_min) * tile_w, (i - x_min + 1) * tile_w
            y0, y1 = (j - y_min) * tile_h, (j - y_min + 1) * tile_h

            # Add the tile to the visual field
            visual_field[x0:x1, y0:y1] = tile

    if convert_to_float:
      visual_field = visual_field.astype(np.float32) / 255.0
    else:
      visual_field = visual_field.astype(np.uint8)

    return visual_field
