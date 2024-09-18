# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import torch
import random
import numpy as np
import jax

from typing import Optional, Union, Sequence
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from IPython.display import clear_output

# Import gem packages
from gem.primitives import GridworldEnv

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #


def visual_field(
    world: np.ndarray,
    color_map,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    channels: int = 5,
    return_rgb=False,
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        channels: (int) defines the size of the visualization. By default, 5 channels. \n
        return_rgb: (bool) Whether to return the image as a plottable RGB image.

    Returns:
        An np.ndarray of C x H x W, determined either by the world size or the vision size.
    """
    C = channels  # Number of channels
    if return_rgb:
        C = 3
        colors = color_map(C)

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack([np.zeros_like(world, dtype=np.float64) for _ in range(C)], axis=0)[
        :, :, :, 0
    ]
    # Get wall appearance from the world object (just pick the first wall object for simplicity)
    wall_appearance = GridworldEnv.get_entities_(world, "Wall")[0].appearance

    # Iterate through the world and assign the appearance of the object at that location
    for index, _ in np.ndenumerate(world[:, :, 0]):
        H, W = index  # Get the coordinates
        # Return visualization image
        if return_rgb:
            new[:, H, W] = colors[world[H, W, 0].kind]
        else:
            new[:, H, W] = world[H, W, 0].appearance

    # If no location, return the full visual field
    if location is None:
        if return_rgb:
            return new.astype(np.uint8).transpose(
                (1, 2, 0)
            )  # Pyplot wants inputs organized in H x W x C
        else:
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
                np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2]),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (world.shape[0] // 2 - vision, world.shape[0] // 2 + vision + 1)
        crop_w = (world.shape[1] // 2 - vision, world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index  # Get the coordinates
            if np.isnan(
                x
            ):  # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                if return_rgb:
                    new[index] = colors["Wall"][C]
                else:
                    new[index] = wall_appearance[C]

        # Return the agent's sliced observation space
        k = (
            0
            if not hasattr(world[location], "direction")
            else world[location].direction % 4
        )

        if return_rgb:
            new = new.astype(np.uint8).transpose((1, 2, 0))
            # ==rotate==#
            new = jax.numpy.rot90(new, k=k)
            # ==========#
            return new
        else:
            new = new.astype(np.float64)
            # ==rotate==#
            new = np.rot90(new, k=k, axes=(1, 2)).copy()
            # ==========#
            return new


def visual_field_multilayer(
    world: np.ndarray,
    color_map,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    channels: int = 5,
    return_rgb=False,
) -> np.ndarray:
    """
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        channels: (int) defines the size of the visualization. By default, 5 channels. \n
        return_rgb: (bool) Whether to return the image as a plottable RGB image.

    Returns:
        An np.ndarray of C x H x W, determined either by the world size or the vision size.
    """
    C = channels  # Number of channels
    if return_rgb:
        C = 3
        colors = color_map(C)

    # Create an array of equivalent shape to the world map, with C appearance channels
    new = np.stack(
        [np.zeros_like(world, dtype=np.float64) for _ in range(C)], axis=0
    ).squeeze()
    # Get wall appearance from the world object (just pick the first wall object for simplicity)
    wall_appearance = GridworldEnv.get_entities_(world, "Wall")[0].appearance

    # Iterate through the world and assign the appearance of the object at that location
    for layer in range(world.shape[-1]):
        for index, _ in np.ndenumerate(world[:, :, layer]):
            H, W = index  # Get the coordinates
            # Return visualization image
            if return_rgb:
                if world.shape[-1] > 1:
                    new[:, H, W, layer] = colors[world[H, W, layer].kind]
                else:
                    new[:, H, W] = colors[world[H, W, layer].kind]
            else:
                if world.shape[-1] > 1:
                    new[:, H, W, layer] = world[H, W, layer].appearance
                else:
                    new[:, H, W] = world[H, W, layer].appearance
    else:
        new = np.sum(new, axis=-1)

    # If no location, return the full visual field
    if location is None:
        if return_rgb:
            return new.astype(np.uint8).transpose(
                (1, 2, 0)
            )  # Pyplot wants inputs organized in H x W x C
        else:
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
                np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2]),
            )
        )

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
        new = shift(array=new, shift=shift_dims, cval=np.nan)

        # Set up the dimensions of the array to crop
        crop_h = (world.shape[0] // 2 - vision, world.shape[0] // 2 + vision + 1)
        crop_w = (world.shape[1] // 2 - vision, world.shape[1] // 2 + vision + 1)
        # Crop the array to the selected dimensions
        new = new[:, slice(*crop_h), slice(*crop_w)]

        for index, x in np.ndenumerate(new):
            C, H, W = index  # Get the coordinates
            if np.isnan(
                x
            ):  # If the current location was outside the shift coordinates...
                # ...replace the appearance with the wall appearance in this channel.
                if return_rgb:
                    new[index] = colors["Wall"][C]
                else:
                    new[index] = wall_appearance[C]
        # Return the agent's sliced observation space
        if return_rgb:
            new = new.astype(np.uint8).transpose((1, 2, 0))
            # ==rotate==#
            new = jax.numpy.rot90(new, k=world[location].direction % 4)
            # ==========#
            return new
        else:
            new = new.astype(np.float64)
            # ==rotate==#
            new = np.rot90(new, k=world[location].direction % 4, axes=(1, 2)).copy()
            # ==========#
            return new


def visual_field_sprite(
    world: np.ndarray,
    location: Optional[ArrayLike] = None,
    vision: Optional[int] = None,
    tile_size: Sequence[int] = [16, 16],
) -> list[np.ndarray]:
    """
    Create an agent visual field of size (2k + 1, 2k + 1) tiles

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        tile_size: (Sequence[int]) defines the size of the sprites. Default: 16 x 16.

    Returns:
        A list of np.ndarrays of C x H x W, determined either by the world size or the vision size.
    """

    # get wall sprite
    wall_sprite = GridworldEnv.get_entities_(world, "Wall")[0].sprite

    # If no location is provided, place the location on the centre of the map with enough space to see the whole world map
    if location is None:
        location = (world.shape[0] // 2, world.shape[1] // 2)
        # Use the largest location dimension to ensure that the entire map is visible in the event of a non-square map
        vision_i = location[0]
        vision_j = location[1]
    else:
        vision_i = vision
        vision_j = vision

    # Layer handling...
    # Separate images will be generated per layer. These will be returned as a list and can then be plotted as a list.
    layers = []
    for z in range(world.shape[2]):

        bounds = (
            location[0] - vision_i,
            location[0] + vision_i,
            location[1] - vision_j,
            location[1] + vision_j,
        )

        image_r = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_g = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_b = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )
        image_a = np.zeros(
            ((2 * vision_i + 1) * tile_size[0], (2 * vision_j + 1) * tile_size[1])
        )

        image_i = 0
        image_j = 0

        for i in range(bounds[0], bounds[1] + 1):
            for j in range(bounds[2], bounds[3] + 1):
                if i < 0 or j < 0 or i >= world.shape[0] or j >= world.shape[1]:
                    # Tile is out of bounds, use wall_app
                    tile_image = (
                        Image.open(os.path.expanduser(wall_sprite))
                        .resize(tile_size)
                        .convert("RGBA")
                    )
                else:
                    tile_appearance = world[i, j, z].sprite
                    tile_image = (
                        Image.open(os.path.expanduser(tile_appearance))
                        .resize(tile_size)
                        .convert("RGBA")
                    )

                tile_image_array = np.array(tile_image)
                alpha = tile_image_array[:, :, 3]
                # tile_image_array[alpha == 0, :3] = 0
                image_r[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 0]
                image_g[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 1]
                image_b[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 2]
                image_a[
                    image_i * tile_size[0] : (image_i + 1) * tile_size[0],
                    image_j * tile_size[1] : (image_j + 1) * tile_size[1],
                ] = tile_image_array[:, :, 3]

                image_j += 1
            image_i += 1
            image_j = 0

        # image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        image = np.zeros((image_r.shape[0], image_r.shape[1], 4))
        image[:, :, 0] = image_r
        image[:, :, 1] = image_g
        image[:, :, 2] = image_b
        image[:, :, 3] = image_a
        layers.append(np.asarray(image, dtype=np.uint8))
    return layers


def plot(image: np.ndarray | list[np.ndarray]) -> None:
    """Plot helper function that takes an image or list of layers."""
    if isinstance(image, np.ndarray):
        plt.imshow(image)
        plt.show()
    else:
        for layer in image:
            plt.imshow(layer)
        plt.show()


def image_from_array(image: np.ndarray | list[np.ndarray]) -> Image:
    """Create a PIL image from an single-layer image or list of layers."""
    if isinstance(image, np.ndarray):
        output = Image.fromarray(image, mode="RGBA")
    else:
        output = Image.fromarray(image[0], mode="RGBA")
        for layer in image[1:]:
            next_layer = Image.fromarray(layer, mode="RGBA")
            output.paste(next_layer, (0, 0), mask=next_layer)
    return output


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image.

    NOTE: DO NOT use this with plt.show(), as it will not work and will return a blank image.

    Parameters:
        fig: If in fig, axis format, then fig. If in plt format, then plt.

    Returns:
        img: An image file in PIL format."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def animate(
    frames: Sequence[PngImageFile],
    filename: Union[str, os.PathLike],
    folder: Union[str, os.PathLike],
):
    """
    Take an array of frames and assemble them into a GIF with the given path.

    Parameters:
        frames: the array of frames \n
        filename: A filename to save the images to \n
        folder: The path to save the gif to
    """
    path = folder + filename + ".gif"

    frames[0].save(
        os.path.expanduser(path),
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


# --------------------------- #
# endregion: Visualizations   #
# --------------------------- #

# --------------------------- #
# region: Helper functions    #
# --------------------------- #


def set_seed(seed: int) -> None:
    """
    Sets a seed for replication.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_seed() -> int:
    """
    Generates a random seed

    Returns:
        The value of the seed generated
    """
    seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def shift(array: ArrayLike, shift: Sequence, cval=np.nan):
    """
    Returns copy of array shifted by offset, with fill using constant.

    Parameters:
        array: The array to shift. \n
        shift: A sequence of dimensions equivalent to the array passed
        into the function. \n
        cval: The value to replace any new elements introduced into the
        offset array. By default, replaces them with nan's.
    """
    offset = np.atleast_1d(shift)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    def slice1(o):
        return slice(o, None) if o >= 0 else slice(0, o)

    new_array[tuple(slice1(o) for o in offset)] = array[
        tuple(slice1(-o) for o in offset)
    ]

    for axis, o in enumerate(offset):
        new_array[
            (slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)
        ] = cval

    return new_array


def nearest_2_power(n: int) -> int:
    """
    Computes the next power of 2. Useful for programmatically
    shifting batch and buffer sizes to computationally efficient
    values.
    """

    # Bit shift counter
    bit_shifts = 0

    # If `n` is already a power of 2 (bitwise n & (n - 1)),
    # return `n` (unless n is 0, handled below)
    if n and not (n & (n - 1)):
        return n

    # Otherwise, repeatedly shift `n` rightwards by 1 bit
    # until `n` is 0...
    while n != 0:
        n >>= 1
        bit_shifts += 1

    # ...then left shift 1 by the number of times n was shifted
    return 1 << bit_shifts


def minmax(n: int, minimum: int, maximum: int) -> int:
    """
    Clips an input to a number between the minimum
    and maximum values passed into the function.
    """
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n


def one_hot_encode(value, num_classes):
    # Create a zero array of length num_classes
    one_hot = np.zeros(num_classes)

    # Set the index corresponding to 'value' to 1
    one_hot[value] = 1

    return one_hot


# --------------------------- #
# endregion                   #
# --------------------------- #
