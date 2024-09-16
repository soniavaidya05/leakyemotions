# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

# Import base packages
import numpy as np
import random

# Import gem-specific packages
from gem.primitives import GridworldEnv
from examples.trucks.entities import (
    Object,
    Wall,
    EmptyObject
)
from examples.trucks.agents import (
    Agent
)
from examples.trucks.config import Cfg
from examples.trucks.utils import color_map


# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

class FoodTrucks(GridworldEnv):
    '''
    Food trucks environment.
    '''
    def __init__(
        self,
        cfg: Cfg,
        agents: list[Agent],
        entities: list[Object]
    ):
        self.cfg = cfg
        self.channels = cfg.env.channels
        self.colors = color_map(self.channels)
        self.full_mdp = cfg.env.full_mdp
        self.baker_mode = cfg.env.baker_mode
        self.agents = agents
        self.trucks = entities
        super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(self.colors['EmptyObject']))
        self.populate()

    # --------------------------- #
    # region: initialization      #
    # --------------------------- #

    def populate(self):
        '''
        Populate the world with objects
        '''
        # First, create the walls
        for index in np.ndindex(self.world.shape):
            H, W, L = index
            # If the index is the first or last, replace the location with a wall
            if H in [0, self.height - 1] or W in [0, self.width - 1]:
                self.world[index] = Wall(appearance=self.colors['Wall'])

        # Normal mode: randomly placed in the environment
        if not self.baker_mode:
            candidate_locs = [index for index in np.ndindex(self.world.shape) if not self.world[index].kind == 'Wall']
            loc_index = np.random.choice(len(candidate_locs), size = 4, replace = False)
            locs = [candidate_locs[i] for i in loc_index]

        # Baker mode: wall placed along the centre and trucks in each corner
        else:
            wall_locs = [(self.height // 2, x, 0) for x in range(4,self.height - 1)]
            for wall_loc in wall_locs:
                self.world[wall_loc] = Wall(appearance = self.colors['Wall'])
            
            candidate_locs = [(1, 1, 0), (1, self.height - 2, 0), (self.height - 2, 1, 0)]
            loc_index = np.random.choice(len(candidate_locs), size = 3, replace = False)
            locs = [candidate_locs[i] for i in loc_index]
            # the agent is always added last so the agent is in the bottom right corner
            locs = np.vstack((locs, [(self.height - 2, self.height - 2, 0)]))

        # Place the trucks and agents
        random.shuffle(self.trucks)
        for i in range(len(locs)):
            loc = tuple(locs[i])
            if i < 3:
                self.world[loc] = self.trucks[i]
                self.trucks[i].location = loc
            else:
                self.world[loc] = self.agents[0]
                self.agents[0].location = loc

    # --------------------------- #
    # endregion: initialization   #
    # --------------------------- #

    def reset(self):
        '''
        Reset the environment.
        '''
        self.create_world()
        self.populate()