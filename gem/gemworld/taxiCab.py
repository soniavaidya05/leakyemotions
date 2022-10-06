from gem.environment.elements.taxiCab_elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from models.perception import agent_visualfield
import random

from utils import find_moveables, find_instance


class TaxiCabEnv:
    def __init__(
        self,
        height=10,
        width=10,
        layers=1,
        defaultObject=EmptyObject(),
    ):
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate()
        self.insert_walls(self.height, self.width)

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

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
        image_r = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_g = np.random.random((self.world.shape[0], self.world.shape[1]))
        image_b = np.random.random((self.world.shape[0], self.world.shape[1]))

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                image_r[i, j] = self.world[i, j, layer].appearence[0]
                image_g[i, j] = self.world[i, j, layer].appearence[1]
                image_b[i, j] = self.world[i, j, layer].appearence[2]

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

        img = agent_visualfield(self.world, moveList[0], k=4)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        taxi_cab_start1 = round(self.world.shape[0] / 2)
        taxi_cab_start2 = round(self.world.shape[1] / 2)
        taxi_start = (taxi_cab_start1, taxi_cab_start2, 0)
        self.world[taxi_start] = TaxiCab(0)

        valid = False
        while not valid:
            loc1 = random.randint(1, self.world.shape[0] - 2)
            loc2 = random.randint(1, self.world.shape[1] - 2)
            passenger_start = (loc1, loc2, 0)
            if taxi_start != passenger_start:
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
            state = models[holdObject.policy].pov(self.world, loc, holdObject)
            action = models[holdObject.policy].take_action([state, epsilon])

            """
            Updates the world given an action
            """
            (
                self.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = holdObject.transition(self.world, models, action, loc)
        else:
            reward = 0
            next_state = state

        additional_output = []

        return state, action, reward, next_state, done, new_loc, additional_output
