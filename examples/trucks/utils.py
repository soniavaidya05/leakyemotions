# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import torch
import numpy as np

from typing import Union
from IPython.display import clear_output

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: Visualizations      #
# --------------------------- #

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
# endregion: Visualizations   #
# --------------------------- #