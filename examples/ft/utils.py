import numpy as np
from numpy.typing import ArrayLike

from examples.ft.gridworld import GridworldEnv

def visual_field(world: np.ndarray,
        location: ArrayLike = None,
        vision: int = None,
        channels: int = 5,
        return_rgb = False
        ) -> np.ndarray:
    '''
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels

    Returns: 
        An np.ndarray of C x H x W, determined either by the world size or the vision size.
    '''
    C = channels # Number of channels
    if return_rgb:
        C = 3
        colors = color_map(C)

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack([np.zeros_like(world, dtype=np.float64) for _ in range(C)], axis = 0).squeeze()
    # Get wall appearance from the world object (just pick the first wall object for simplicity)
    wall_appearance = GridworldEnv.get_entities_(world, 'Wall')[0].appearance

    # Iterate through the world and assign the appearance of the object at that location
    for index, _ in np.ndenumerate(world[:, :, 0]):
        H, W = index # Get the coordinates
        # Return visualization image
        if return_rgb:
            new[:, H, W] = colors[world[H, W, 0].kind]
        else:
            new[:, H, W] = world[H, W, 0].appearance

    # If no location, return the full visual field
    if location is None:
        if return_rgb:
            return new.astype(np.uint8).transpose((1, 2, 0))
        else:
            return new.astype(np.uint8)
    
    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.hstack((
            [0], # The first dimension is zero, because the channels are not shifted
            np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2])
        ))

        # Shift the array
        new = shift(
            array=new,
            shift=shift_dims,
            cval = np.nan
        )

        # Set up the dimensions of the array to crop
        crop_h = (world.shape[0] // 2 - vision, world.shape[0] // 2 + vision + 1)
        crop_w = (world.shape[1] // 2 - vision, world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index # Get the coordinates
            if np.isnan(x): # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                if return_rgb:
                    new[index] = colors['Wall'][C]
                else:
                    new[index] = wall_appearance[C]

        if return_rgb:
            return new.astype(np.uint8).transpose((1, 2, 0))
        else:
            return new.astype(np.uint8)

def add_models(agents, models):
    for agent, model in zip(agents, models):
        agent.model = model

# region: Helper functions
def shift(array, 
          shift, 
          cval = np.nan):
    """Returns copy of array shifted by offset, with fill using constant."""
    offset = np.atleast_1d(shift)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = (
        array[tuple(slice1(-o) for o in offset)])

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis +
                (slice(0, o) if o >= 0 else slice(o, None),)] = cval

    return new_array

def color_map(channels):
    '''
    Generates a color map for the food truck environment.
    '''
    if channels > 4:
        colors = {
            'EmptyObject': [0 for _ in range(channels)],
            'Agent': [255 if x == 0 else 0 for x in range(channels)],
            'Wall': [255 if x == 1 else 0 for x in range(channels)],
            'koreanTruck': [255 if x == 2 else 0 for x in range(channels)],
            'lebaneseTruck': [255 if x == 3 else 0 for x in range(channels)],
            'mexicanTruck': [255 if x == 4 else 0 for x in range(channels)]
        }
    else:
        colors = {
            'EmptyObject': [0.0, 0.0, 0.0],
            'Agent': [200.0, 200.0, 200.0],
            'Wall': [50.0, 50.0, 50.0],
            'koreanTruck': [0.0, 0.0, 255.0],
            'lebaneseTruck': [0.0, 255.0, 0.0],
            'mexicanTruck': [255.0, 0.0, 0.0]
        }
    return colors

# endregion

    
