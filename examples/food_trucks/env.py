from examples.food_trucks.elements import (
    Agent,
    Truck,
    KoreanTruck,
    LebaneseTruck,
    MexicanTruck,
    EmptyObject,
    Wall,
)

import numpy as np
import matplotlib.pyplot as plt
from gem.models.perception_singlePixel_categories import agent_visualfield
from gem.models.perception import agent_visualfield as agent_visualfield2

from gem.utils import find_instance, one_hot
from IPython.display import clear_output
import torch


class FoodTrucks:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        tile_size=(1, 1),
        truck_prefs = (10, 5, -5),
        baker_mode = False,
        one_hot = True,
        vision = 5,
        full_mdp = False
    ):
        self.one_hot = one_hot
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.baker_mode = baker_mode
        self.truck_prefs = truck_prefs
        self.vision = vision
        self.full_mdp = full_mdp
        self.colors = self.color_map()
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(truck_prefs)
        self.insert_walls(self.height, self.width)
        self.tile_size = tile_size

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        
        self.defaultObject.appearance = self.colors['empty_color']
        self.world = np.full((height, width, layers), self.defaultObject)

    def color_map(self):
        if self.one_hot:
            colors = {
                'agent_color': [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0], # index 2
                'wall_color': [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0], # index 1
                'empty_color': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                'korean_color': [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0], # index 3
                'lebanese_color': [0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0], # index 4
                'mexican_color': [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0] # index 5
            }
        else:
            colors = {
                'agent_color': [200.0, 200.0, 200.0],
                'wall_color': [50.0, 50.0, 50.0],
                'empty_color': [0.0, 0.0, 0.0],
                'korean_color': [0.0, 0.0, 255.0],
                'lebanese_color': [0.0, 255.0, 0.0],
                'mexican_color': [255.0, 0.0, 0.0]
            }
        return colors


    def populate(self, truck_prefs):
        """
        Populates the game board with elements
        TODO: test whether the probabilities above are working
        """

        if truck_prefs is not None:
            korean_truck, lebanese_truck, mexican_truck = truck_prefs
        else:
            korean_truck, lebanese_truck, mexican_truck = self.truck_prefs

        if self.baker_mode == False:
            candidate_locs = [index for index in np.ndindex(self.world.shape) if not isinstance(self.world[index[0], index[1], index[2]], Wall) and not index[2] != 0]
            loc_index = np.random.choice(len(candidate_locs), size = 4, replace = False)
            locs = [candidate_locs[i] for i in loc_index]
        else:
            candidate_locs = [(1, 1, 0), (1, self.height - 2, 0), (self.height - 2, 1, 0)]
            loc_index = np.random.choice(len(candidate_locs), size = 3, replace = False)
            # the agent is always added last so the agent is in the bottom right corner
            locs = [candidate_locs[i] for i in loc_index]
            locs = np.vstack((locs, [(self.height - 2, self.height - 2, 0)]))
            wall_locs = [(int(np.floor(self.height/2)), x, 0) for x in range(4,self.height - 1)]
            for wall_loc in wall_locs:
                self.world[wall_loc[0], wall_loc[1], wall_loc[2]] = Wall(color = self.colors['wall_color'])

        self.world[locs[0][0], locs[0][1], locs[0][2]] = KoreanTruck(korean_truck, color = self.colors['korean_color'])
        self.world[locs[1][0], locs[1][1], locs[1][2]] = LebaneseTruck(lebanese_truck, color = self.colors['lebanese_color'])
        self.world[locs[2][0], locs[2][1], locs[2][2]] = MexicanTruck(mexican_truck, color = self.colors['mexican_color'])
        self.world[locs[3][0], locs[3][1], locs[3][2]] = Agent(model = 0, 
                                                               color = self.colors['agent_color'],
                                                               vision = self.vision)

    def insert_walls(self, height, width):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        for i in range(height):
            self.world[0, i, 0] = Wall(color = self.colors['wall_color'])
            self.world[height - 1, i, 0] = Wall(color = self.colors['wall_color'])
            self.world[i, 0, 0] = Wall(color = self.colors['wall_color'])
            self.world[i, height - 1, 0] = Wall(color = self.colors['wall_color'])

    def reset_env(self):
        """
        Resets the environment and repopulates it
        """
        self.create_world(self.height, self.width, self.layers)
        self.populate(self.truck_prefs)
        self.insert_walls(self.height, self.width)

    def pov(self, location, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        TODO: to get better flexibility, this code should be moved to env
        """

        previous_state = self.world[location].episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)

            if not self.full_mdp:
                if self.one_hot:
                    img = agent_visualfield(
                        self.world,
                        loc,
                        k=self.world[location].vision,
                        wall_app=[0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        num_channels=7,
                    )
                else:
                    img = agent_visualfield2(
                        self.world,
                        loc,
                        k=self.world[location].vision,
                        out_of_bounds_colour=self.colors['wall_color']
                )
            # Full mdp version: see whole map, centred on middle pixel
            else:
                loc = (self.height // 2, self.width // 2, layer)
                if self.one_hot:
                    img = agent_visualfield(
                        self.world,
                        loc,
                        k=self.height // 2,
                        wall_app=[0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        num_channels=7,
                    )
                else:
                    img = agent_visualfield2(
                        self.world,
                        loc,
                        k=self.height // 2,
                        out_of_bounds_colour=self.colors['wall_color']
                )   
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)

        if len(inventory) > 0:
            """
            Loops through each additional piece of information and places into one layer
            """
            inventory_var = torch.tensor([])
            for item in range(len(inventory)):
                tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
                inventory_var = torch.cat((inventory_var, tmp), dim=0)
            inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
            state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state
    
    def pov_action(self, location):
        '''
        Create an agent action frames 
        '''
        n_actions = 4
        agent = self.world[location]
        # Get the number of timesteps to append
        B, T, C, H, W = agent.episode_memory[-1][1][0].size()
        previous_actions = torch.zeros((B, T, n_actions))
        for i in range(T):
            action = agent.episode_memory[-4+i][1][1]
            previous_actions[B, i, :] = one_hot(n_actions, pos = action)
        return previous_actions
        
