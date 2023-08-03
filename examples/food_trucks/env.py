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
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception_singlePixel_categories import agent_visualfield

from gem.utils import find_instance
from IPython.display import clear_output
import torch


class RPG:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        tile_size=(1, 1),
        truck_prefs = (10, 5, -5),
        baker_mode = False
    ):
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.baker_mode = baker_mode
        self.truck_prefs = truck_prefs
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(truck_prefs)
        self.insert_walls(self.height, self.width)
        self.tile_size = tile_size

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

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
                self.world[wall_loc[0], wall_loc[1], wall_loc[2]] = Wall()

        self.world[locs[0][0], locs[0][1], locs[0][2]] = KoreanTruck(korean_truck, [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0])
        self.world[locs[1][0], locs[1][1], locs[1][2]] = LebaneseTruck(lebanese_truck, [0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0])
        self.world[locs[2][0], locs[2][1], locs[2][2]] = MexicanTruck(mexican_truck, [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0])
        self.world[locs[3][0], locs[3][1], locs[3][2]] = Agent(model = 0)

    def insert_walls(self, height, width):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        for i in range(height):
            self.world[0, i, 0] = Wall()
            self.world[height - 1, i, 0] = Wall()
            self.world[i, 0, 0] = Wall()
            self.world[i, height - 1, 0] = Wall()

    def reset_env(
        self, height=15, width=15, layers=1
    ):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(self.truck_prefs)
        self.insert_walls(height, width)

    def plot(self, layer):  # is this defined in the master?
        """
        Creates an RGB image of the whole world
        """
        image_r = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_g = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_b = np.random.random((self.world.shape[0], self.world.shape[1]))

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                image_r[i, j] = self.world[i, j, layer].appearance[0]
                image_g[i, j] = self.world[i, j, layer].appearance[1]
                image_b[i, j] = self.world[i, j, layer].appearance[2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image

    def init_elements(self):
        """
        Creates objects that survive from game to game
        """
        self.emptyObject = EmptyObject()
        self.walls = Wall()

    def game_test(self, layer=0):
        """
        Prints one frame to check game instance parameters
        """
        image = self.plot(layer)

        moveList = find_instance(self.world, "neural_network")

        img = agent_visualfield(
            self.world,
            moveList[0],
            k=4,
            wall_app=[0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            num_channels=7,
        )

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

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
            img = agent_visualfield(
                self.world,
                loc,
                k=self.world[location].vision,
                wall_app=[0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                num_channels=7,
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

    # def step(self, models, loc, epsilon=0.85, device=None):
    #     """
    #     This is an example script for an alternative step function
    #     It does not account for the fact that an agent can die before
    #     it's next turn in the moveList. If that can be solved, this
    #     may be preferable to the above function as it is more like openAI gym

    #     The solution may come from the agent.died() function if we can get that to work

    #     location = (i, j, 0)

    #     Uasge:
    #         for i, j, k = agents
    #             location = (i, j, k)
    #             state, action, reward, next_state, done, additional_output = env.stepSingle(models, (0, 0, 0), epsilon)
    #             env.world[0, 0, 0].updateMemory(state, action, reward, next_state, done, additional_output)
    #         env.WorldUpdate()

    #     """
    #     holdObject = self.world[loc]
    #     device = models[holdObject.policy].device

    #     if holdObject.kind != "deadAgent":
    #         """
    #         This is where the agent will make a decision
    #         If done this way, the pov statement may be about to be part of the action
    #         Since they are both part of the same class

    #         if going for this, the pov statement needs to know about location rather than separate
    #         i and j variables
    #         """
    #         state = models[holdObject.policy].pov(self.world, loc, holdObject)
    #         params = (state.to(device), epsilon, None)
    #         action, init_rnn_state = models[holdObject.policy].take_action(params)

    #     if holdObject.has_transitions == True:
    #         """
    #         Updates the world given an action
    #         TODO: does this need self.world in here, or can it be figured out by passing self?
    #         """
    #         (
    #             self.world,
    #             reward,
    #             next_state,
    #             done,
    #             new_loc,
    #         ) = holdObject.transition(self, models, action, loc)
    #     else:
    #         reward = 0
    #         next_state = state

    #     additional_output = []

    #     return state, action, reward, next_state, done, new_loc, additional_output
