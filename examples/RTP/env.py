from examples.RTP.elements import (
    Agent,
    Gem,
    EmptyObject,
    Wall,
)

import numpy as np
import matplotlib.pyplot as plt
from gem.models.perception_singlePixel_categories import agent_visualfield
import random

from gem.utils import find_instance
import torch


class RTP:
    def __init__(
        self,
        height=15,
        width=15,
        layers=1,
        contextual=True,
        tile_size=(1, 1),
        appearance_size=20,
    ):
        self.app_size = appearance_size
        self.height = height
        self.width = width
        self.layers = layers
        self.defaultObject = EmptyObject(appearance_size)
        self.tile_size = (tile_size,)
        self.contextual = contextual
        self.create_world()
        self.insert_walls()
        self.create_people()
        self.populate()

    def create_world(self):
        """
        Creates a world of the specified size with a default object
        """
        self.world = np.full((self.height, self.width, self.layers), self.defaultObject)

    def insert_walls(self):
        """
        Inserts walls into the world.
        """
        for i in range(self.height):
            for j in range(self.width):
                self.world[0, j, 0] = Wall(self.app_size)
                self.world[self.height - 1, j, 0] = Wall(self.app_size)
                self.world[i, 0, 0] = Wall(self.app_size)
                self.world[i, self.width - 1, 0] = Wall(self.app_size)

    def create_people(self, num_people=100):
        """
        Create a new list of people.
        NOTE: This list is MAINTAINED ACROSS GAMES,
        and is only generated on environment initialization
        """
        self.person_list = []

        for person in range(num_people):
            # Individual appearance
            individuation = [random.random() * 255 for i in range(8)]
            zeroes = [0 for i in range(8)]
            color = np.random.choice([0, 1])

            # Group id depends on appearance index 2 or 3
            # -------------------------------------------
            # Group 0 is mostly miners
            if color == 0:
                image_color = [0.0, 0.0, 255.0, 0.0]
                if random.random() < 0.75:
                    rock = 1
                    wood = 0
                else:
                    rock = 0
                    wood = 1
            # Group 1 is 50/50 choppers and miners
            if color == 1:
                image_color = [0.0, 0.0, 0.0, 255.0]
                if random.random() < 0.25:
                    rock = 1
                    wood = 0
                else:
                    rock = 0
                    wood = 1
            # Appearance includes 4 object ids and 8 individuation characteristics
            # Trimmed to max appearance size
            app = np.array([image_color + individuation + zeroes][0][: self.app_size])
            info = (person, app, [wood, rock], 0, 0)
            self.person_list.append(info)

    def populate(self, change=False, masked_attitude=False):
        """
        Populates the game board with elements
        """

        n = 20  # the number of people to learn per board
        random_numbers = random.sample(range(len(self.person_list)), n)

        # Create a list of candidate locations
        candidate_locs = [
            index
            for index in np.ndindex(self.world.shape)
            if not isinstance(self.world[index[0], index[1], index[2]], Wall)
            and not index[2] != 0
        ]

        # Choose n + 1 random locations without replacement
        loc_index = np.random.choice(len(candidate_locs), size=n + 1, replace=False)
        locs = [candidate_locs[i] for i in loc_index]

        # The last loc is used to place the agent, the rest are used for the people
        for person in range(n):
            app = self.person_list[random_numbers[person]][1]
            reward = self.person_list[random_numbers[person]][2]
            self.world[locs[person]] = Gem(reward, app, self.app_size)

        # Place the agent
        self.world[locs[20]] = Agent(0, self.app_size)

    def reset_env(
        self,
        change=False,
        masked_attitude=True,
    ):
        """
        Resets the environment and repopulates it
        """
        self.create_world()
        self.insert_walls()
        self.populate(change=change)

    def reset_appearance(self, loc):
        """
        Reset the last 5 values for the environment appearance
        appearance[-5]: agent's wood value
        appearance[-4]: agent's stone value
        appearance[-3:]: used for attitude models
        """
        for i in range(self.height):
            for j in range(self.width):
                self.world[i, j, 0].appearance[-5] = self.world[loc].wood * 255.0
                self.world[i, j, 0].appearance[-4] = self.world[loc].stone * 255.0
                self.world[i, j, 0].appearance[-3] = 0.0
                self.world[i, j, 0].appearance[-2] = 0.0
                self.world[i, j, 0].appearance[-1] = 0.0

    def pov(self, location, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        TODO: to get better flexibility, this code should be moved to env
        """

        previous_state = self.world[location].episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]
        wall_colour = np.zeros(self.app_size)
        wall_colour[1] = 255.0
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
                wall_app=wall_colour,
                num_channels=self.app_size,
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
