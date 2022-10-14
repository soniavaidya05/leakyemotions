# from gem.environment.elements import Agent, EmptyObject, Gem, Wall, Wolf
from gem.environment.elements.agent import Agent
from gem.environment.elements.element import EmptyObject, Wall
from gem.environment.elements.gem import Gem
from gem.environment.elements.wolf import Wolf
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.perception import agent_visualfield


class WolfsAndGems:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        gem1p=0.03,
        gem2p=0.04,
        wolf1p=0.03,
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
        self.insert_walls()

    def create_world(self, height=15, width=15, layers=1):
        # self.world = np.full((self.height, self.width, self.layers), self.defaultObject)
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(
        self, height=15, width=15, layers=1, gem1p=0.110, gem2p=0.04, wolf1p=0.005
    ):
        self.create_world(height, width, layers)
        self.populate(gem1p, gem2p, wolf1p)
        self.insert_walls()
        # needed because the previous version was resetting the replay buffer
        # in the reset we should be able to make a bigger or smaller world
        # right now the game is stuck in 15x15, and will want to start increasing
        # the size of the world as the agents learn

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
                if self.world[i, j, 0].static == 0:
                    moveList.append([i, j])

        img = agent_visualfield(self.world, (moveList[0][0], moveList[0][1]), k=4)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self, gem1p=0.03, gem2p=0.04, wolf1p=0.035, agent1p=0.005):
        """
        Populates the game board with elements
        TODO: test whether the probabilites above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2, 3, 4],
                    p=[
                        gem1p,
                        gem2p,
                        wolf1p,
                        agent1p,
                        1 - gem2p - gem1p - wolf1p - agent1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = Gem(5, [0.0, 255.0, 0.0])
                if obj == 1:
                    self.world[i, j, 0] = Gem(15, [255.0, 255.0, 0.0])
                if obj == 2:
                    self.world[i, j, 0] = Wolf(1)
                if obj == 3:
                    self.world[i, j, 0] = Agent(0)

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

    def insert_walls(self):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        # wall = Wall()
        for i in range(self.height):
            self.world[0, i, 0] = Wall()
            self.world[self.height - 1, i, 0] = Wall()
            self.world[i, 0, 0] = Wall()
            self.world[i, self.height - 1, 0] = Wall()
