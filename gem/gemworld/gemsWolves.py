from gem.environment.elements.agent import Agent
from gem.environment.elements.element import EmptyObject, Wall
from gem.environment.elements.gem import Gem
from gem.environment.elements.wolf import Wolf
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from models.perception import agent_visualfield
import random

from utils import find_moveables, find_instance
import torch

class WolfsAndGems:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        gem1p=0.110,
        gem2p=0.04,
        wolf1p=0.005,
    ):
        self.gem1p = gem1p
        self.gem2p = gem2p
        self.wolf1p = wolf1p
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.gem1p, self.gem2p, self.wolf1p)
        self.insert_walls(self.height, self.width)

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(
        self, height=15, width=15, layers=1, gem1p=0.110, gem2p=0.04, wolf1p=0.005
    ):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(gem1p, gem2p, wolf1p)
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

        img = agent_visualfield(self.world, moveList[0], k=4)

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
            img = agent_visualfield(self.world, loc, self.world[location].vision)
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

    def populate(self, gem1p=0.115, gem2p=0.06, wolf1p=0.005):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2, 3],
                    p=[
                        gem1p,
                        gem2p,
                        wolf1p,
                        1 - gem2p - gem1p - wolf1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = Gem(5, [0.0, 255.0, 0.0])
                if obj == 1:
                    self.world[i, j, 0] = Gem(15, [255.0, 255.0, 0.0])
                if obj == 2:
                    self.world[i, j, 0] = Wolf(1)

        cBal = np.random.choice([0, 1])
        if cBal == 0:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = Agent(0)
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = Agent(0)
        if cBal == 1:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = Agent(0)
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = Agent(0)

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
        This is an example script for an alternative step function
        It does not account for the fact that an agent can die before
        it's next turn in the moveList. If that can be solved, this
        may be preferable to the above function as it is more like openAI gym

        The solution may come from the agent.died() function if we can get that to work

        location = (i, j, 0)

        Uasge:
            for i, j, k = agents
                location = (i, j, k)
                state, action, reward, next_state, done, additional_output = env.stepSingle(models, (0, 0, 0), epsilon)
                env.world[0, 0, 0].updateMemory(state, action, reward, next_state, done, additional_output)
            env.WorldUpdate()

        """
        holdObject = self.world[loc]
        device = models[holdObject.policy].device
        


        if holdObject.kind != "deadAgent":
            """
            This is where the agent will make a decision
            If done this way, the pov statement may be about to be part of the action
            Since they are both part of the same class

            if going for this, the pov statement needs to know about location rather than separate
            i and j variables
            """
            state = models[holdObject.policy].pov(self.world, loc, holdObject)
            action = models[holdObject.policy].take_action([state.to(device), epsilon])

        if holdObject.has_transitions == True:
            """
            Updates the world given an action
            TODO: does this need self.world in here, or can it be figured out by passing self?
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
