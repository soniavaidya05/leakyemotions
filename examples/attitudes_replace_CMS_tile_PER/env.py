from examples.attitudes_replace_CMS_tile_PER.elements import (
    Agent,
    Gem,
    EmptyObject,
    Wall,
)

import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception_singlePixel_categories import agent_visualfield
import random

from gem.utils import find_moveables, find_instance
import torch
import pdb

class RPG:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        defaultObject=EmptyObject(),
        gem1p=0.110,
        gem2p=0.04,
        wolf1p=0.005,
        tile_size=(1, 1),
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
        self.tile_size = tile_size
        self.gem1_value = ((np.random.random() - 0.5) * 30) + 1
        self.gem2_value = ((np.random.random() - 0.5) * 30) + 1
        self.gem3_value = ((np.random.random() - 0.5) * 30) + 1
        self.gem1_appearance = [
            np.random.random() * 255,
            np.random.random() * 255,
            np.random.random() * 255,
            0,
            0,
        ]
        self.gem2_appearance = [
            np.random.random() * 255,
            np.random.random() * 255,
            np.random.random() * 255,
            0,
            0,
        ]
        self.gem3_appearance = [
            np.random.random() * 255,
            np.random.random() * 255,
            np.random.random() * 255,
            0,
            0,
        ]
        self.populate(self.gem1p, self.gem2p, self.wolf1p)
        self.gem_values = [self.gem1_value, self.gem2_value, self.gem3_value]
        self.insert_walls(self.height, self.width)
        self.change_gem_values()

    def rotate_rgb(self, rgb, angle):
        # Convert angle to radians
        angle = np.radians(angle)

        # Define the rotation matrix
        rotation_matrix = np.array(
            [
                [
                    np.cos(angle) + 1 / 3 * (1 - np.cos(angle)),
                    1 / 3 * (1 - np.cos(angle)) - np.sqrt(1 / 3) * np.sin(angle),
                    1 / 3 * (1 - np.cos(angle)) + np.sqrt(1 / 3) * np.sin(angle),
                ],
                [
                    1 / 3 * (1 - np.cos(angle)) + np.sqrt(1 / 3) * np.sin(angle),
                    np.cos(angle) + 1 / 3 * (1 - np.cos(angle)),
                    1 / 3 * (1 - np.cos(angle)) - np.sqrt(1 / 3) * np.sin(angle),
                ],
                [
                    1 / 3 * (1 - np.cos(angle)) - np.sqrt(1 / 3) * np.sin(angle),
                    1 / 3 * (1 - np.cos(angle)) + np.sqrt(1 / 3) * np.sin(angle),
                    np.cos(angle) + 1 / 3 * (1 - np.cos(angle)),
                ],
            ]
        )

        # Multiply the rotation matrix with the RGB values
        rotated_rgb = rotation_matrix.dot(rgb)

        # Clamp the values between 0 and 255
        rotated_rgb = np.clip(rotated_rgb, 0, 255)

        return tuple(rotated_rgb)

    def create_world(self, height=15, width=15, layers=1):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((height, width, layers), self.defaultObject)

    def add_object(self):
        """
        Adds an object to the world
        """

        gem1p = 0.03
        gem2p = 0.03
        gem3p = 0.03

        obj = np.random.choice(
            [0, 1, 2, 3],
            p=[
                gem1p,
                gem2p,
                gem3p,
                1 - gem2p - gem1p - gem3p,
            ],
        )
        placed = False

        counter = 0

        while not placed:
            object_location1 = np.random.choice(np.arange(1, self.world.shape[1] - 1))
            object_location2 = np.random.choice(np.arange(1, self.world.shape[1] - 1))
            object_location = (object_location1, object_location2, 0)
            if self.world[object_location1, object_location2, 0].kind == "empty":
                if obj == 0:
                    self.world[object_location] = Gem(
                        self.gem1_value, self.gem1_appearance
                    )
                if obj == 1:
                    self.world[object_location] = Gem(
                        self.gem2_value, self.gem2_appearance
                    )
                if obj == 2:
                    self.world[object_location] = Gem(
                        self.gem3_value, self.gem3_appearance
                    )
                placed = True
            counter += 1
            if counter > 100:
                print("Could not place object")
                break

    def change_gem_values(self, new_values="Shuffled", new_colours="Shuffled"):
        if new_values == "Random":
            val1 = np.random.random() * 15
            val2 = np.random.random() * 15
            val3 = np.random.random() * 15

            flip_neg = np.random.choice([0, 1, 2])
            if flip_neg == 0:
                val1 = (val2 + val3) * -0.5
            if flip_neg == 1:
                val2 = (val1 + val3) * -0.5
            if flip_neg == 2:
                val3 = (val1 + val2) * -0.5

            self.gem1_value = val1
            self.gem2_value = val2
            self.gem3_value = val3

        if new_values == "Shuffled":
            gem_values = np.random.choice([0, 1, 2])

            if gem_values == 0:
                self.gem1_value = 15
                gem_values2 = np.random.choice([0, 1])
                if gem_values2 == 0:
                    self.gem2_value = 5
                    self.gem3_value = -5
                else:
                    self.gem2_value = -5
                    self.gem3_value = 5
            elif gem_values == 1:
                self.gem1_value = 5
                gem_values2 = np.random.choice([0, 1])
                if gem_values2 == 0:
                    self.gem2_value = 15
                    self.gem3_value = -5
                else:
                    self.gem2_value = -5
                    self.gem3_value = 15
            elif gem_values == 2:
                self.gem1_value = -5
                gem_values2 = np.random.choice([0, 1])
                if gem_values2 == 0:
                    self.gem2_value = -5
                    self.gem3_value = 15
                else:
                    self.gem2_value = 15
                    self.gem3_value = -5

        if new_colours == "Random":
            self.gem1_appearance = [
                np.random.random() * 255,
                np.random.random() * 255,
                np.random.random() * 255,
                0,
                0,
            ]
            self.gem2_appearance = [
                np.random.random() * 255,
                np.random.random() * 255,
                np.random.random() * 255,
                0,
                0,
            ]
            self.gem3_appearance = [
                np.random.random() * 255,
                np.random.random() * 255,
                np.random.random() * 255,
                0,
                0,
            ]

        if new_colours == "Shuffled":
            color1 = [200, 50, 100]
            color2 = [50, 100, 200]
            color3 = [100, 200, 50]

            gem_colours = np.random.choice([0, 1, 2])

            if gem_colours == 0:
                self.gem1_appearance = color1
                gem_colours2 = np.random.choice([0, 1])
                if gem_colours2 == 0:
                    self.gem2_appearance = color2
                    self.gem3_appearance = color3
                else:
                    self.gem2_appearance = color3
                    self.gem3_appearance = color2
            elif gem_colours == 1:
                self.gem1_appearance = color2
                gem_colours2 = np.random.choice([0, 1])
                if gem_colours2 == 0:
                    self.gem2_appearance = color1
                    self.gem3_appearance = color3
                else:
                    self.gem2_appearance = color3
                    self.gem3_appearance = color1
            elif gem_colours == 2:
                self.gem1_appearance = color3
                gem_colours2 = np.random.choice([0, 1])
                if gem_colours2 == 0:
                    self.gem2_appearance = color1
                    self.gem3_appearance = color2
                else:
                    self.gem2_appearance = color2
                    self.gem3_appearance = color1

        if new_colours == "Correlated":
            g1 = tuple(self.gem1_appearance[0:3])
            g2 = tuple(self.gem2_appearance[0:3])
            g3 = tuple(self.gem3_appearance[0:3])

            # g1 = (g1t[0], g1t[1], g1t[2])

            direction = np.random.choice([0, 1])
            if direction == 0:
                g1n = self.rotate_rgb(g1, 45)
            else:
                g1n = self.rotate_rgb(g1, -45)

            direction = np.random.choice([0, 1])
            if direction == 0:
                g2n = self.rotate_rgb(g2, 45)
            else:
                g2n = self.rotate_rgb(g2, -45)

            direction = np.random.choice([0, 1])
            if direction == 0:
                g3n = self.rotate_rgb(g3, 45)
            else:
                g3n = self.rotate_rgb(g3, -45)

            self.gem1_appearance[0] = g1n[0]
            self.gem2_appearance[0] = g2n[0]
            self.gem3_appearance[0] = g3n[0]

            self.gem1_appearance[1] = g1n[1]
            self.gem2_appearance[1] = g2n[1]
            self.gem3_appearance[1] = g3n[1]

            self.gem1_appearance[2] = g1n[2]
            self.gem2_appearance[2] = g2n[2]
            self.gem3_appearance[2] = g3n[2]

            self.gem1_appearance[3] = 0
            self.gem2_appearance[3] = 0
            self.gem3_appearance[3] = 0

            self.gem1_appearance[4] = 0
            self.gem2_appearance[4] = 0
            self.gem3_appearance[4] = 0

        self.gem_values = [self.gem1_value, self.gem2_value, self.gem3_value]

    def reset_env(
        self,
        height=15,
        width=15,
        layers=1,
        gem1p=0.110,
        gem2p=0.04,
        gem3p=0.005,
        change=False,
        masked_attitude=True,
    ):
        """
        Resets the environment and repopulates it
        """
        self.create_world(height, width, layers)
        self.populate(gem1p, gem2p, gem3p, change=change)
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

        green = [0, 255, 0]
        wall_app = [
            green, green, green, green, green,
            green, green, green, green, green,
            green, green, green, green, green,
            green, green, green, green, green,
            green, green, green, green, green,
        ]

        img = agent_visualfield(
            self.world,
            moveList[0],
            k=4,
            wall_app=wall_app,
            num_channels=75,
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

            green = [0, 255, 0]
            wall_app = [
                green, green, green, green, green,
                green, green, green, green, green,
                green, green, green, green, green,
                green, green, green, green, green,
                green, green, green, green, green,
            ]

            img = agent_visualfield(
                self.world,
                loc,
                k=self.world[location].vision,
                wall_app=wall_app,
                num_channels=75,
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

    def populate(
        self, gem1p=0.115, gem2p=0.06, gem3p=0.005, change=False, masked_attitude=False
    ):
        """
        Populates the game board with elements
        TODO: test whether the probabilities above are working
        """

        for i in range(self.world.shape[0]):
            for j in range(self.world.shape[1]):
                obj = np.random.choice(
                    [0, 1, 2, 3],
                    p=[
                        gem1p,
                        gem2p,
                        gem3p,
                        1 - gem2p - gem1p - gem3p,
                    ],
                )

                if obj == 0:
                    self.world[i, j, 0] = Gem(self.gem1_value, self.gem1_appearance)
                if obj == 1:
                    self.world[i, j, 0] = Gem(self.gem2_value, self.gem2_appearance)
                if obj == 2:
                    self.world[i, j, 0] = Gem(self.gem3_value, self.gem3_appearance)

        player1_location1 = np.random.choice(np.arange(1, self.world.shape[1] - 1))
        player1_location2 = np.random.choice(np.arange(1, self.world.shape[1] - 1))

        player2_location1 = 0
        player2_location2 = 0

        while player1_location1 != player2_location1:
            player2_location1 = np.random.choice(np.arange(1, self.world.shape[1] - 1))
            player2_location2 = np.random.choice(np.arange(1, self.world.shape[1] - 1))

        self.world[player1_location1, player1_location2, 0] = Agent(0)
        self.world[player2_location1, player2_location2, 0] = Agent(0)

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



def agent_visualfield(
    world, location, k, wall_app, num_channels
):
    """
    Create an agent visual field of size (2k + 1, 2k + 1) pixels
    Layer = location[2] and layer in the else are added to this function
    """

    if len(location) > 2:
        layer = location[2]
    else:
        layer = 0

    bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)
    # instantiate image
    images = []

    for channel in range(num_channels):
        image = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
        images.append(image)

    for i in range(bounds[0], bounds[1] + 1):
        for j in range(bounds[2], bounds[3] + 1):
            # while outside the world array index...
            if i < 0 or j < 0 or i >= world.shape[0] - 1 or j >= world.shape[1]:
                # image has shape bounds[1] - bounds[0], bounds[3] - bounds[2]
                # visual appearance = wall
                flat = torch.FloatTensor(wall_app).flatten()
                for channel in range(num_channels):
                    images[channel][i - bounds[0], j - bounds[2]] = flat[channel]

            else:
                flat = torch.FloatTensor(world[i, j, layer].appearance).flatten()
                for channel in range(num_channels):
                    images[channel][i - bounds[0], j - bounds[2]] = flat[channel]

    # Composite image by interlacing the red, green, and blue channels, or one hots
    state = np.dstack(tuple(images))
    return state