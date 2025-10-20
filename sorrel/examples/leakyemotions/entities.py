"""The entities for the leaky emotions project."""

# begin imports
from pathlib import Path
from scipy.stats import norm

import numpy as np

from sorrel.entities import Entity
from sorrel.worlds import Gridworld
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld

# end imports

class Bush(Entity[LeakyEmotionsWorld]):
    """An entity that represents a bush in the leakyemotions environment."""   

    def __init__(self, location=None, ripe_num=0, lifespan=15):
        super().__init__()
        self.value = 1 
        self.ripeness = ripe_num
        self.lifespan = lifespan
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/bush_new.png"
        self.kind="Bush"
        self.values = self.value_function()
    
    def transition(self, world: LeakyEmotionsWorld):
        self.ripeness += 1
        if self.ripeness >= self.lifespan * (1/3):
            self.sprite = Path(__file__).parent / "./assets/bush.png"
        if self.ripeness >= self.lifespan * (2/3):
            self.sprite = Path(__file__).parent / "./assets/bush_old.png"
        if self.ripeness > self.lifespan:
            world.remove(self.location)
            world.add(self.location, Grass(bush_lifespan=self.lifespan))
        else:
            self.determine_value()
        # return world

    def value_function(self):
        x = np.arange(0,self.lifespan+1)
        # values normalized with maximum value of 5
        return 5 * norm.pdf(x, loc=self.lifespan / 2, scale=self.lifespan / 6) /\
              norm.pdf(x, loc=self.lifespan / 2, scale=self.lifespan / 6).max()
        
    def determine_value(self):
        self.value = self.values[self.ripeness] 

class Wall(Entity[Gridworld]):
    """An entity that represents a wall in the leakyemotions environment."""
    
    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = Path(__file__).parent / "./assets/wall.png"

class Grass(Entity[LeakyEmotionsWorld]):
    """An entity that represents a block of grass in the treasurehunt environment."""

    def __init__(self, bush_lifespan=15):
        super().__init__()
        # We technically don't need to make Grass passable here since it's on a different layer from Agent
        self.passable = True
        self.has_transitions = True
        self.sprite = Path(__file__).parent / "./assets/grass.png"
        self.bush_lifespan = bush_lifespan

    def transition(self, world: LeakyEmotionsWorld):
        """Grass can randomly spawn into Bushes based on the item spawn probabilities dictated in the evironment."""
    
        if (  # NOTE: If the spawn prob is too high, the environment gets overrun
            np.random.random() < world.spawn_prob
        ):
            world.add(self.location, Bush(lifespan=self.bush_lifespan))

class EmptyEntity(Entity[Gridworld]):
    """An entity that represents an empty space in the leakyemotions environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.has_transitions = True  # EmptyEntity can transition into Bushes
        self.sprite = Path(__file__).parent / "./assets/empty.png"
