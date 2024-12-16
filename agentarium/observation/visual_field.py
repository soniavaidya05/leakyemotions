# Import base packages
import numpy as np

# Import gem packages
from agentarium.primitives import GridworldEnv
from agentarium.utils.helpers import shift


def visual_field(
    env: GridworldEnv,
    entity_map: dict[str, list[float]],
    vision: int | None = None,
    location: tuple | None = None,
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        env (GridworldEnv): The environment to visualize.
        entity_map (dict[str, list[float]]): The mapping 
        between objects and visual appearance.
        vision (Optional, int): The agent's visual field
        radius. If none, the entire environment.
        location: (Optional, tuple): The location to centre
        the visual field on. If none, the entire environment.

    Returns:
        np.ndarray: An array of shape 
        `(2 * vision + 1, 2 * vision + 1, env.layers)`.
        Or if vision is None: 
        `(env.height, env.width, env.layers)`.
    """
    # Get the number of channels used by the model.
    num_channels = len(list(entity_map.values())[0])

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack(
        [np.zeros_like(env.world, dtype=np.float64) for _ in range(num_channels)], axis=0
    ).squeeze()

    # Iterate through the world and assign the appearance of the object at that location
    for layer in range(env.world.shape[-1]):
        for index, _ in np.ndenumerate(env.world[:, :, layer]):
            H, W = index  # Get the coordinates
            # Return visualization image
            if env.world.shape[-1] > 1:
                new[:, H, W, layer] = entity_map[env.world[H, W, layer].kind]
            else:
                new[:, H, W] = entity_map[env.world[H, W, layer].kind]
    else:
        new = np.sum(new, axis=-1)

    # If no location, return the full visual field
    if location is None:
        return new.astype(np.float64)

    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.hstack(
            (
                [
                    0
                ],  # The first dimension is zero, because the channels are not shifted
                np.subtract([env.world.shape[0] // 2, env.world.shape[1] // 2], location[0:2]),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (env.world.shape[0] // 2 - vision, env.world.shape[0] // 2 + vision + 1)
        crop_w = (env.world.shape[1] // 2 - vision, env.world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index  # Get the coordinates
            if np.isnan(
                x
            ):  # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                new[index] = entity_map["Wall"][C]
        # Return the agent's sliced observation space
        new = new.astype(np.float64)
        # ==rotate==#
        new = np.rot90(new, k=env.world[location].direction % 4, axes=(1, 2)).copy()
        # ==========#
        return new
