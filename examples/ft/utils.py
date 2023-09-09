# --------------- #
# region: Imports #
# Import base packages
import torch
import random
import numpy as np

from typing import Optional, Union
from numpy.typing import ArrayLike
from PIL import Image
from IPython.display import clear_output

# Import gem packages
from examples.ft.gridworld import GridworldEnv
# endregion       #
# --------------- #

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
        channels: (int, Optional) defines the size of the visualization. By default, 5 channels. \n
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
            return new.astype(np.uint8).transpose((1, 2, 0))
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

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

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
            *flags
        ) -> None:
        '''
        Take the results from a given epoch (epoch #, turn #, loss, and reward) 
        and return a formatted string that can be printed to the command line.
        '''
        
        if 'jupyter-mode' in flags:
            clear_output(wait = True)
            print(f'╔═════════════╦═══════════╦═════════════╦═════════════╗')
            print(f'║ Epoch: {str(self.epochs[-1]).rjust(4)} ║ Turn: {str(self.turns[-1]).rjust(3)} ║ Loss: {str(self.losses[-1]).rjust(5)} ║ Reward: {str(self.rewards[-1]).rjust(3)} ║')
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

# --------------------------- #
# endregion                   #
# --------------------------- #
    
