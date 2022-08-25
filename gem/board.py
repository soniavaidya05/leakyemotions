from abc import ABC

# I assume the below is an automatic addition and not what we want
#from turtle import width
from gem.environment.elements import Agent, EmptyObject, Gem, Wall, Wolf
import numpy as np
from abc import ABC, abstractmethod
from astropy.visualization import make_lupton_rgb



class Board(ABC):
    def __init__(self, height, width, layers, defaultObject):
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObjectClass = EmptyObject()
        self.create_world()

    def create_world(self):
        self.world = np.full(self.height, self.width, self.layers, self.defaultObjectClass)

    def insert_walls(self):
        """
        Inserts walls into the world.
        Assumes that the world is square - fixme.
        """
        wall = Wall()
        for i in range(self.height):
            self.world[0, i, 0] = wall
            self.world[self.height - 1, i, 0] = wall
            self.world[i, 0, 0] = wall
            self.world[i, self.height - 1, 0] = wall

    def find_instance(self, world, kind):
        """
        TODO: consider implementing type hinting for kind param, etc.
        """
        instList = []
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.layers):
                    if world[i, j, k].kind == kind:
                        instList.append(world[i, j, k])
        return instList

    def plot(self, layer=0):
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

    @abstractmethod
    def populate(self):
        pass


class WolfsAndGems(Board):
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
        super().__init__(height, width, layers, defaultObject)
        self.insert_walls()
        self.gem1p = gem1p
        self.gem2p = gem2p
        self.wolf1p = wolf1p

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

    def populate(self):
        for i in range(self.height):
            for j in range(self.width):
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
            self.world[round(self.height / 2), round(self.width / 2), 0] = self.agent1
            self.world[
                round(self.height / 2) + 1, round(self.width / 2) - 1, 0
            ] = self.agent1
        if cBal == 1:
            self.world[round(self.height / 2), round(self.width / 2), 0] = self.agent1
            self.world[
                round(self.height / 2) + 1, round(self.width / 2) - 1, 0
            ] = self.agent1

