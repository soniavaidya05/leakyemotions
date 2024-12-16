"""The entities for treasurehunt, a simple example for the purpose of a tutorial."""

# TODO: first file to write!

import numpy as np

from agentarium.primitives import Entity
# TODO: test if this (& type hinting the environment in code) would cause an issue
from examples.treasurehunt.env import TreasurehuntEnv


class EmptySpace(Entity):
    """An entity that represents an empty space in the treasurehunt environment."""

    def __init__(self, appearance):
        super().__init__(appearance)
        self.passable = True  # Agents can enter EmptySpaces
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/sand.png"
        )

    def transition(self, env):
        """
        EmptySpaces can randomly spawn into Gems based on the item spawn probabilities dictated in the environmnet.
        """
        if (
            np.random.random() < env.spawn_prob
        ):  # NOTE: If this rate is too high, the environment gets overrun
            env.add(self.location, Gem(gem_appearance))


class Wall(Entity):
    """An entity that represents a wall in the treasurehunt environment."""

    def __init__(self, appearance):
        super().__init__(appearance)
        self.value = -1  # Walls penalize contact
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/wall.png"
        )


class Gem(Entity):
    """An entity that represents a gem in the treasurehunt environment."""

    def __init__(self, appearance, gem_value):
        super().__init__(appearance)
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/gem.png"
        )
