from gem.environment.elements import Agent, EmptyObject, Gem, Wall, Wolf
import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.perception import agentVisualField


class WolfsAndGems:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        gem1p=0.115,
        gem2p=0.06,
        wolf1p=0.005,
    ):
        self.gem1p = gem1p
        self.gem2p = gem2p
        self.wolf1p = wolf1p
        self.height = (height,)
        self.width = (width,)
        self.layers = (layers,)
        self.defaultObject = defaultObject
        self.create_world()
        self.init_elements()
        self.populate()
        self.insert_walls()

    def create_world(self, height=15, width=15, layers=1):
        # self.world = np.full((self.height, self.width, self.layers), self.defaultObject)
        self.world = np.full((height, width, layers), self.defaultObject)

    def reset_env(self):
        self.create_world()
        self.populate()
        self.insert_walls()
        # needed because the previous version was resetting the replay buffer

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
        self.agent1 = Agent(0)
        self.wolf1 = Wolf(1)
        self.gem1 = Gem(5, [0.0, 255.0, 0.0])
        self.gem2 = Gem(15, [255.0, 255.0, 0.0])
        self.emptyObject = EmptyObject()
        self.walls = Wall()

    # below make it so that it only puts objects in the non wall parts.
    # this may need to have a parameter that indicates whether things can be
    # on the edges or not

    def gameTest(self, layer=0):
        image = self.plot(layer)

        moveList = []
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                if self.world[i, j, 0].static == 0:
                    moveList.append([i, j])

        img = agentVisualField(self.world, (moveList[0][0], moveList[0][1]), k=4)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()

    def populate(self):
        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2, 3],
                    p=[
                        self.gem1p,
                        self.gem2p,
                        self.wolf1p,
                        1 - self.gem2p - self.gem1p - self.wolf1p,
                    ],
                )
                if obj == 0:
                    self.world[i, j, 0] = self.gem1
                if obj == 1:
                    self.world[i, j, 0] = self.gem2
                if obj == 2:
                    self.world[i, j, 0] = self.wolf1

        cBal = np.random.choice([0, 1])
        if cBal == 0:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = self.agent1
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = self.agent1
        if cBal == 1:
            self.world[
                round(self.world.shape[0] / 2), round(self.world.shape[1] / 2), 0
            ] = self.agent1
            self.world[
                round(self.world.shape[0] / 2) + 1,
                round(self.world.shape[1] / 2) - 1,
                0,
            ] = self.agent1

    def insert_walls(self):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        wall = Wall()
        for i in range(15):
            self.world[0, i, 0] = wall
            self.world[15 - 1, i, 0] = wall
            self.world[i, 0, 0] = wall
            self.world[i, 15 - 1, 0] = wall
