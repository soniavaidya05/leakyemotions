# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import torch
import random
import numpy as np

from typing import Optional, Union, Sequence
from numpy.typing import ArrayLike
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from IPython.display import clear_output

# Import gem packages
from examples.ft.gridworld import GridworldEnv

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #

def visual_field(world: np.ndarray,
        location: Optional[ArrayLike] = None,
        vision: Optional[int] = None,
        channels: int = 5,
        return_rgb = False
        ) -> np.ndarray:
    '''
    Visualize the world.

    Parameters:
        location: (ArrayLike, Optional) defines the location to centre the visualization on \n
        vision: (int, Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels \n
        channels: (int) defines the size of the visualization. By default, 5 channels. \n
        return_rgb: (bool) Whether to return the image as a plottable RGB image.

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
            return new.astype(np.uint8).transpose((1, 2, 0)) # Pyplot wants inputs organized in H x W x C
        else:
            return new.astype(np.float64)
    
    # Otherwise...
    else:
        # The centrepoint for the shift array is defined by the centrepoint on the main array
        # E.g. the centrepoint for a 9x9 array is (4, 4). So, the shift array for the location
        # (1, 6) is (3, -2): left three, up two.
        shift_dims = np.hstack((
            [0], # The first dimension is zero, because the channels are not shifted
            np.subtract([world.shape[0] // 2, world.shape[1] // 2], location[0:2])
        ))

        # Shift the array.
        # Replace coordinates outside the map with nans. We will fix this later.
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

        # Return the agent's sliced observation space
        if return_rgb:
            return new.astype(np.uint8).transpose((1, 2, 0)) 
        else:
            return new.astype(np.float64)

def color_map(channels: int) -> dict:
    '''
    Generates a color map for the food truck environment.

    Parameters:
        channels: the number of appearance channels in the environment

    Return:
        A dict of object-color mappings
    '''
    if channels > 4:
        colors = {
            'EmptyObject': [0 for _ in range(channels)],
            'Agent': [255 if x == 0 else 0 for x in range(channels)],
            'Wall': [255 if x == 1 else 0 for x in range(channels)],
            'korean': [255 if x == 2 else 0 for x in range(channels)],
            'lebanese': [255 if x == 3 else 0 for x in range(channels)],
            'mexican': [255 if x == 4 else 0 for x in range(channels)]
        }
    else:
        colors = {
            'EmptyObject': [0.0, 0.0, 0.0],
            'Agent': [200.0, 200.0, 200.0],
            'Wall': [50.0, 50.0, 50.0],
            'korean': [0.0, 0.0, 255.0],
            'lebanese': [0.0, 255.0, 0.0],
            'mexican': [255.0, 0.0, 0.0]
        }
    return colors

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def animate(
        frames: Sequence[PngImageFile], 
        filename: Union[str, os.PathLike], 
        folder: Union[str, os.PathLike] = '/Users/rgelpi/Documents/GitHub/transformers/examples/ft/data/'):
    '''
    Take an array of frames and assemble them into a GIF with the given path.

    Parameters:
        frames: the array of frames \n
        filename: A filename to save the images to \n
        folder: The path to save the gif to
    '''
    path = folder + filename + '.gif'

    frames[0].save(path, format = 'GIF', append_images = frames[1:], save_all = True, duration = 100, loop = 0)

# --------------------------- #
# endregion: Visualizations   #
# --------------------------- #

# --------------------------- #
# region: Helper functions    #
# --------------------------- #

def set_seed(seed: int) -> None:
    '''
    Sets a seed for replication.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def random_seed() -> int:
    '''
    Generates a random seed

    Returns:
        The value of the seed generated
    '''
    seed = random.randint(0,10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def shift(array: ArrayLike, 
          shift: Sequence, 
          cval = np.nan):
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

    new_array[tuple(slice1(o) for o in offset)] = (
        array[tuple(slice1(-o) for o in offset)])

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis +
                (slice(0, o) if o >= 0 else slice(o, None),)] = cval

    return new_array

def nearest_2_power(n: int) -> int:
    '''
    Computes the next power of 2. Useful for programmatically 
    shifting batch and buffer sizes to computationally efficient
    values.
    '''
    
    # Bit shift counter
    bit_shifts = 0
 
    # If `n` is already a power of 2 (bitwise n & (n - 1)),
    # return `n` (unless n is 0, handled below)
    if (n and not(n & (n - 1))):
        return n
    
    # Otherwise, repeatedly shift `n` rightwards by 1 bit
    # until `n` is 0...
    while( n != 0):
        n >>= 1
        bit_shifts += 1
     
    # ...then left shift 1 by the number of times n was shifted
    return 1 << bit_shifts

def minmax(
    n: int,
    minimum: int,
    maximum: int
) -> int:
    '''
    Clips an input to a number between the minimum
    and maximum values passed into the function.
    '''
    if n < minimum:
        return minimum
    elif n > maximum:
        return maximum
    else:
        return n    

# --------------------------- #
# endregion                   #
# --------------------------- #

# --------------------------- #
# region: Game data storage   #
# --------------------------- #

class GameVars:
    '''
    Container for storing game variables.
    '''
    def __init__(self, max_epochs):
        self.epochs = []
        self.turns = []
        self.losses = []
        self.rewards = []
        self.max_epochs = max_epochs

    def clear(self):
        '''
        Clear the game variables.
        '''
        del self.epochs[:]
        del self.turns[:]
        del self.losses[:]
        del self.rewards[:]
    
    def record_turn(
        self,
        epoch: int,
        turn: int,
        loss: Union[float, torch.Tensor],
        reward: Union[int, float, torch.Tensor]
    ):
        '''
        Record a game turn.
        '''
        self.epochs.append(epoch)
        self.turns.append(turn)
        self.losses.append(np.round(loss, 3))
        self.rewards.append(reward)

    def pretty_print(
            self,
            *flags,
            **kwargs
        ) -> None:
        '''
        Take the results from a given epoch (epoch #, turn #, loss, and reward) 
        and return a formatted string that can be printed to the command line.

        If `jupyter-mode` is passed in as a flag, variables need to be passed 
        in with the `kwargs`.
        '''
        
        if 'jupyter-mode' in flags:
            assert all(key in kwargs.keys() for key in ('epoch', 'turn', 'reward')), 'Jupyter mode requires the current epoch, turn, and reward to be passed in as kwargs.'
            clear_output(wait = True)
            print(f'╔═════════════╦═══════════╦═════════════╦═════════════╗')
            print(f'║ Epoch: {str(kwargs["epoch"]).rjust(4)} ║ Turn: {str(kwargs["turn"]).rjust(3)} ║ Loss: {str("None").rjust(5)} ║ Reward: {str(kwargs["reward"]).rjust(3)} ║')
            print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝')
        else:
            if self.epochs[-1] == 0:
                print(f'╔═════════════╦═══════════╦═════════════╦═════════════╗')
            else:
                print(f'╠═════════════╬═══════════╬═════════════╬═════════════╣')
            if True:
                print(f'║ Epoch: {str(self.epochs[-1]).rjust(4)} ║ Turn: {str(self.turns[-1]).rjust(3)} ║ Loss: {str(self.losses[-1]).rjust(5)} ║ Reward: {str(self.rewards[-1]).rjust(3)} ║')
                print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝',end='\r')
            if self.epochs[-1] == self.max_epochs - 1:
                print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝')

    def __repr__(self):
        return f'{self.__class__.__name__}(n_games={len(self.epochs)})'

# --------------------------- #
# endregion                   #
# --------------------------- #
    
