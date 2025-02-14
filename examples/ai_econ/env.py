import random

import numpy as np

from agentarium.environments import GridworldEnv
from agentarium.observation.observation import ObservationSpec
from examples.ai_econ.agents import Buyer, Seller
from examples.ai_econ.entities import (EmptyEntity, Land, StoneNode, Wall,
                                       WoodNode)


class AIEconomist(GridworldEnv):
    """
    AI Economist environment.
    """

    def __init__(self, cfg):
        layers = 3
        default_entity = EmptyEntity()
        super().__init__(cfg.env.height, cfg.env.width, layers, default_entity)

        self.cfg = cfg
        self.entity_list = [
            "EmptyEntity",
            "Wall",
            "Land",
            "WoodNode",
            "StoneNode",
            "Seller",
            "Buyer",
        ]
        self.woodcutters = []
        self.stonecutters = []
        self.markets = []

        # TODO: based on the size of the environment, have a hard limit on the number of agents

        self.num_buyers = cfg.agent.buyer.num
        self.majority_percentage = cfg.agent.buyer.majority_percentage
        self.num_sellers = cfg.agent.seller.num

        self.max_turns = cfg.experiment.max_turns

        self.create_agents()
        self.populate()

    def create_agents(self):
        """
        Creates the agents for this environment.
        Appearances are placeholders for now; 0 for woodcutters, 1 for stonecutters, and 2 for markets.
        """
        for i in range(0, self.num_sellers):
            self.woodcutters.append(
                Seller(
                    self.cfg,
                    appearance=0,
                    is_majority=(i >= self.num_sellers * self.majority_percentage),
                    is_woodcutter=True,
                    observation_spec=ObservationSpec(self.entity_list),
                )
            )
            self.stonecutters.append(
                Seller(
                    self.cfg,
                    appearance=1,
                    is_majority=(i >= self.num_sellers * self.majority_percentage),
                    is_woodcutter=False,
                    observation_spec=ObservationSpec(self.entity_list),
                )
            )

        for _ in range(0, self.num_buyers):
            self.markets.append(
                Buyer(
                    self.cfg,
                    appearance=2,
                    observation_spec=ObservationSpec(self.entity_list),
                )
            )

    def populate(self):
        """
        Populate the treasurehunt world by creating walls, then randomly spawning 1 gem and 1 agent.
        Note that every space is already filled with EmptyEntity as part of super().__init__().

        Note: work in progress to make this change with the size of the world!
        For now, the environment is somewhat hard coded and only works with height=51 and width=51.
        """
        for index in np.ndindex(self.world.shape):
            y, x, z = index

            # walls
            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            if (x == 15 or x == 31) and (y in range(15, 20) or y in range(31, 36)):
                # Add top & bottom walls around the market area
                self.add(index, Wall())
            if (y == 15 or y == 31) and (x in range(15, 20) or x in range(31, 36)):
                # Add left & right walls around the market area
                self.add(index, Wall())

            # resource nodes, which are on the bottom layer
            if z == 0:
                # wood nodes
                if x in range(1, 15) and y in range(1, 15):
                    self.add(index, WoodNode(self.cfg))
                elif x in range(36, 50) and y in range(36, 50):
                    self.add(index, WoodNode(self.cfg))
                # stone nodes
                elif x in range(1, 15) and y in range(36, 50):
                    self.add(index, StoneNode(self.cfg))
                elif x in range(36, 50) and y in range(1, 15):
                    self.add(index, StoneNode(self.cfg))
                # land
                else:
                    self.add(index, Land())

        # finished filling in entities; spawn agents in a separate method cause this one is getting long
        self.place_agents()

    def place_agents(self):
        """
        Places the agents into the environment.
        """
        north_spawn_area = []
        south_spawn_area = []
        east_spawn_area = []
        west_spawn_area = []
        for y, x in np.ndindex(self.height, self.width):
            if x in range(20, 31):
                if y in range(1, 15):
                    north_spawn_area.append((y, x, 1))
                if y in range(36, 50):
                    south_spawn_area.append((y, x, 1))
            if y in range(20, 31):
                if x in range(1, 15):
                    west_spawn_area.append((y, x, 1))
                if x in range(36, 50):
                    east_spawn_area.append((y, x, 1))

        # woodcutters: north area & south area
        woodcutters_spawn_locations = random.sample(
            north_spawn_area, self.num_sellers // 2
        ) + random.sample(south_spawn_area, self.num_sellers // 2)
        random.shuffle(woodcutters_spawn_locations)
        # stonecutters: east area & west area
        stonecutters_spawn_locations = random.sample(
            east_spawn_area, self.num_sellers // 2
        ) + random.sample(west_spawn_area, self.num_sellers // 2)
        random.shuffle(stonecutters_spawn_locations)

        for woodcutter, woodcutter_location in zip(
            self.woodcutters, woodcutters_spawn_locations
        ):
            self.add(woodcutter_location, woodcutter)
        for stonecutter, stonecutter_location in zip(
            self.stonecutters, stonecutters_spawn_locations
        ):
            self.add(stonecutter_location, stonecutter)

    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.populate()
        for woodcutter in self.woodcutters:
            woodcutter.reset()
        for stonecutter in self.stonecutters:
            stonecutter.reset()
        for market in self.markets:
            market.reset()
