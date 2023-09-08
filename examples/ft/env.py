# --------------------------------- #
# region: Imports                   #
# --------------------------------- #
from examples.ft.gridworld import GridworldEnv
from examples.ft.entities import (
    EmptyObject,
    Wall,
    Truck
)
from examples.ft.agents import Agent
import numpy as np

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

class FoodTrucks(GridworldEnv):
    '''
    Food trucks environment.
    '''
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        self.channels = cfg.env.channels
        self.truck_prefs = cfg.env.truck_prefs
        self.full_mdp = cfg.env.full_mdp
        self.baker_mode = cfg.env.baker_mode
        self.color_map()
        super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(self.colors['empty']))
        self.populate()

    # --------------------------- #
    # region: initialization      #
    # --------------------------- #
    
    def color_map(self):
        '''
        Generates a color map for the food truck environment.
        '''
        if self.channels > 4:
            self.colors = {
                'empty': [0 for _ in range(self.channels)],
                'agent': [255 if x == 0 else 0 for x in range(self.channels)],
                'wall': [255 if x == 1 else 0 for x in range(self.channels)],
                'korean': [255 if x == 2 else 0 for x in range(self.channels)],
                'lebanese': [255 if x == 3 else 0 for x in range(self.channels)],
                'mexican': [255 if x == 4 else 0 for x in range(self.channels)]
            }
        else:
            self.colors = {
                'empty': [0.0, 0.0, 0.0],
                'agent': [200.0, 200.0, 200.0],
                'wall': [50.0, 50.0, 50.0],
                'korean': [0.0, 0.0, 255.0],
                'lebanese': [0.0, 255.0, 0.0],
                'mexican': [255.0, 0.0, 0.0]
            }

    def populate(self):
        '''
        Populate the world with objects
        '''
        # First, create the walls
        for index in np.ndindex(self.world.shape):
            H, W, L = index
            # If the index is the first or last, replace the location with a wall
            if H in [0, self.height - 1] or W in [0, self.width - 1]:
                self.world[index] = Wall(color=self.colors['wall'])

        # Normal mode: randomly placed in the environment
        if not self.baker_mode:
            candidate_locs = [index for index in np.ndindex(self.world.shape) if not self.world[index].kind == 'wall']
            loc_index = np.random.choice(len(candidate_locs), size = 4, replace = False)
            locs = [candidate_locs[i] for i in loc_index]

        # Baker mode: wall placed along the centre and trucks in each corner
        else:
            wall_locs = [(self.height // 2, x, 0) for x in range(4,self.height - 1)]
            for wall_loc in wall_locs:
                self.world[wall_loc] = Wall(color = self.colors['wall'])
            
            candidate_locs = [(1, 1, 0), (1, self.height - 2, 0), (self.height - 2, 1, 0)]
            loc_index = np.random.choice(len(candidate_locs), size = 3, replace = False)
            locs = [candidate_locs[i] for i in loc_index]
            # the agent is always added last so the agent is in the bottom right corner
            locs = np.vstack((locs, [(self.height - 2, self.height - 2, 0)]))

        # Place the trucks and agents
        trucks = ('korean', 'lebanese', 'mexican')
        for i in range(len(locs)):
            loc = tuple(locs[i])
            if i < 3:
                self.world[loc] = Truck(
                    color=self.colors[trucks[i]],
                    value=self.truck_prefs[i],
                    cuisine=trucks[i])
            else:
                self.world[loc] = Agent(
                    color=self.colors['agent'],
                    model=None,
                    cfg=self.cfg,
                    location=loc
                )

    # --------------------------- #
    # endregion: initialization   #
    # --------------------------- #

    def reset(self):
        self.create_world()
        self.populate()