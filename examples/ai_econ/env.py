import numpy as np

from agentarium.agents import Agent
from agentarium.entities import Entity
from agentarium.environments import GridworldEnv
from agentarium.location import Location, Vector
from examples.ai_econ.entities import EmptyEntity, Wall, Land, StoneNode, WoodNode

class EconEnv(GridworldEnv):

    def __init__(self, height: int, width: int, gem_value: int, spawn_prob: float, max_turns: int, agents: list[Agent]):
        layers = 3
        default_entity = EmptyEntity()
        super().__init__(height, width, layers, default_entity)

        self.gem_value = gem_value
        self.spawn_prob = spawn_prob
        self.agents = agents
        self.max_turns = max_turns

        self.game_score = 0
        self.populate()

    def populate(self):
        """
        Populate the treasurehunt world by creating walls, then randomly spawning 1 gem and 1 agent.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        valid_spawn_locations = []
        wood_spawn_locations = []
        stone_spawn_locations = []

        for index in np.ndindex(self.world.shape):
            y, x, z = index
            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.add(index, Land())
            elif z == 1: # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                if (y / self.height) < 0.33 and (x / self.width) < 0.33:
                    wood_spawn_locations.append(index)
                elif (y / self.height) > 0.66 and (x / self.width) > 0.66:
                    wood_spawn_locations.append(index)
                elif (y / self.height) > 0.66 and (x / self.width) < 0.33:
                    stone_spawn_locations.append(index)
                elif (y / self.height) < 0.33 and (x / self.width) > 0.66:
                    stone_spawn_locations.append(index)
            elif z == 2:
                valid_spawn_locations.append(index)

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.add(loc, agent)

        # Stone and wood locations
        stone_locations_indices = np.random.choice(
            len(stone_spawn_locations), size=len(stone_spawn_locations) // 2, replace=False
        )
        stone_locations = [stone_spawn_locations[i] for i in stone_locations_indices]
        for loc in stone_locations:
            self.add(tuple(loc), StoneNode(renew_chance=self.spawn_prob, renew_amount=self.gem_value))

        wood_locations_indices = np.random.choice(
            len(wood_spawn_locations), size=len(wood_spawn_locations) // 2, replace=False
        )
        wood_locations = [wood_spawn_locations[i] for i in wood_locations_indices]
        for loc in wood_locations:
            self.add(tuple(loc), WoodNode(renew_chance=self.spawn_prob, renew_amount=self.gem_value))

    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.game_score = 0
        self.populate()
        for agent in self.agents:
            agent.reset()