"""The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# TODO: 2nd file to write!

import random
from email.policy import default

# Import base packages
import numpy as np

# Import primitive types
from agentarium.primitives import GridworldEnv
# Import experiment specific classes
from examples.treasurehunt.entities import EmptyEntity, Gem, Wall


# TODO: change appearances of everything
class Treasurehunt(GridworldEnv):
    """
    Treasurehunt environment.
    """

    def __init__(self, height, width, gem_value, spawn_prob, max_turns, agents):
        layers = 2
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
        Note that every space is already filled with EmptySpace as part of super().__init__().
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.shape):
            y, x, z = index

            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            else:
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the initial gem
        gem_location = random.choice(valid_spawn_locations)
        self.add(gem_location, Gem(self.gem_value))

        # remove the gem location from list of valid spawn locations
        valid_spawn_locations.remove(gem_location)
        # spawn the agents
        agent_locations = np.random.choice(
            np.array(valid_spawn_locations), size=len(self.agents), replace=False
        )
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.add(loc, agent)

        # NOTE: random vs. np.random, don't use both yada yada

    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.game_score = 0
        self.populate()
        for agent in self.agents:
            agent.reset(self)
