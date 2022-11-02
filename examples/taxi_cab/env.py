from examples.taxi_cab.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception import agent_visualfield
import random
import torch

from gem.utils import find_moveables, find_instance


class TaxiCabEnv:
    def __init__(
        self,
        height=10,
        width=10,
        layers=1,
        defaultObject=EmptyObject(),
        tile_size=(3, 3)
    ):
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate()
        self.insert_walls(self.height, self.width)
        self.tile_size = tile_size

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), EmptyObject())
        for i in range(height):
            for j in range(width):
                for k in range(layers):
                    self.world[i, j, k] = EmptyObject()

    def reset_env(self, height=10, width=10, layers=1):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate()
        self.insert_walls(height, width)

    def plot(self, layer):  # is this defined in the master?
        """
        Creates an RGB image of the whole world
        """
        return agent_visualfield(self.world, (0, 0, layer), self.tile_size, k=None)

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

        img = agent_visualfield(self.world, moveList[0], self.tile_size, k=4)

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
            img = agent_visualfield(self.world, loc, self.tile_size, k=self.world[location].vision)
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

    def pov_noCNN(self, location, lstm_input):
        """
        This is being used as a scratch area thinking about non-CNN inputs to the models
        we may concat these inputs into a model that can use both scalars and CNNs
        For example, we should be able to have this read in whether a passenger is in the taxi
        """

        previous_state = self.world[location].episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :] = previous_state[:, 1:, :]

        state_now = torch.tensor(lstm_input).unsqueeze(0).unsqueeze(0)
        current_state[:, -1, :] = state_now

        return current_state

    def populate(self):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        taxi_cab_start1 = round(self.world.shape[0] / 2)
        taxi_cab_start2 = round(self.world.shape[1] / 2)
        taxi_start = (taxi_cab_start1, taxi_cab_start2, 0)
        self.world[taxi_start] = TaxiCab(0)
        self.spawn_passenger()


    def spawn_passenger(self):
        """
        Spawns a passenger in a random location
        """
        valid = False
        while not valid:
            loc1 = random.randint(1, self.world.shape[0] - 2)
            loc2 = random.randint(1, self.world.shape[1] - 2)
            passenger_start = (loc1, loc2, 0)
            if isinstance(self.world[passenger_start], EmptyObject):
                valid = True
                self.world[passenger_start] = Passenger(self.world)

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

    def step(self, models, loc, epsilon=0.85):
        """
        This is an example script for an  step function
        """

        if self.world[loc].action_type == "neural_network":

            holdObject = self.world[loc]
            device = models[holdObject.policy].device
            state = self.pov(loc, inventory=[holdObject.has_passenger], layers=[0])
            params = (state.to(device), epsilon, holdObject.init_rnn_state)
            action, init_rnn_state = models[holdObject.policy].take_action(params)
            self.world[loc].init_rnn_state = init_rnn_state
            """
            Updates the world given an action
            """
            (
                self.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = holdObject.transition(self, models, action, loc)
        else:
            reward = 0
            next_state = state

        additional_output = []

        return state, action, reward, next_state, done, new_loc, additional_output
