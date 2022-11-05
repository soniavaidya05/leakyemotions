from examples.ai_economist.elements import (
    Agent,
    Wood,
    Stone,
    House,
    EmptyObject,
    Wall,
)
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception import agent_visualfield

import torch


class AI_Econ:
    def __init__(
        self,
        height=30,
        width=30,
        layers=2,
        defaultObject=EmptyObject(),
        wood1p=0.04,
        stone1p=0.04,
        tile_size=(1, 1)
    ):
        self.wood1p = wood1p
        self.stone1p = stone1p
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = defaultObject
        self.create_world(self.height, self.width, self.layers)
        self.init_elements()
        self.populate(self.wood1p, self.stone1p)
        self.insert_walls(self.height, self.width, self.layers)
        self.wood = 4
        self.stone = 4
        self.tile_size = tile_size

    def create_world(self, height=30, width=30, layers=2):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(self, height=30, width=30, layers=1, wood1p=0.04, stone1p=0.04):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(wood1p, stone1p)
        self.insert_walls(height, width, layers)

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

        moveList = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i, j, layer].static == 0:
                    moveList.append([i, j, layer])

        if len(moveList) > 0:
            img = agent_visualfield(self.world, moveList[0], self.tile_size, k=4)
        else:
            img = image

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self, wood1p=0.04, stone1p=0.04):
        """
        Populates the game board with elements
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2],
                    p=[
                        wood1p,
                        stone1p,
                        1 - wood1p - stone1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = Wood()
                if obj == 1:
                    self.world[i, j, 0] = Stone()

        """
        Quick and dirty population. Should do this with lists instead
        """

        loc = (3, 7, 1)
        apperence1 = (0., 0., 255.0)
        apperence2 = (50., 0., 255.0)
        apperence3 = (0., 50., 255.0)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (3, 4, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (7, 4, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (23, 7, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (23, 4, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (27, 4, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (23, 27, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (23, 24, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (27, 24, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

        loc = (3, 23, 1)
        self.world[loc] = Agent(
            model=0,
            stone_skill=0.9,
            wood_skill=0.25,
            house_skill=0.25,
            appearance=apperence1,
        )
        loc = (3, 27, 1)
        self.world[loc] = Agent(
            model=1,
            stone_skill=0.25,
            wood_skill=0.9,
            house_skill=0.25,
            appearance=apperence2,
        )
        loc = (7, 24, 1)
        self.world[loc] = Agent(
            model=2,
            stone_skill=0.25,
            wood_skill=0.25,
            house_skill=0.9,
            appearance=apperence3,
        )

    def insert_walls(self, height, width, layers):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        for layer in range(layers):

            for i in range(height):
                self.world[0, i, layer] = Wall()
                self.world[height - 1, i, layer] = Wall()
                self.world[i, 0, layer] = Wall()
                self.world[i, height - 1, layer] = Wall()

            # this is a hack to get to look like AI economist
            for i in range(8):
                self.world[14, i, layer] = Wall()
                self.world[i, 14, layer] = Wall()
            for i in range(8):
                self.world[14, height - i - 1, layer] = Wall()
                self.world[height - i - 1, 14, layer] = Wall()


    def pov(self, world, location, holdObject, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)
            img = agent_visualfield(world, loc, self.tile_size, holdObject.vision)
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

    def step(self, models, loc, epsilon=0.85):
        """
        Have the agent take an action
        """
        holdObject = self.world[loc] # TODO: need to see whether holding this constant is still needed
        device = models[holdObject.policy].device

        if holdObject.static != 1:
            """
            This is where the agent will make a decision
            """
            state = self.pov(
                self.world,
                loc,
                holdObject,
                inventory=[self.world[loc].stone, holdObject.wood, holdObject.coin],
                layers=[0, 1],
            )
            action, init_rnn_state = models[holdObject.policy].take_action([state.to(device), epsilon, None])
            self.world[loc].init_rnn_state = init_rnn_state
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
