import numpy as np

def viz(world: np.ndarray,
        location: tuple = None,
        vision: int = None,
        ):
    '''
    Visualize the world.

    Parameters
    ----------
    location: (Optional) defines the location to centre the visualization on
    vision: (Optional) defines the size of the visualization of (2v + 1, 2v + 1) pixels
    '''
    
