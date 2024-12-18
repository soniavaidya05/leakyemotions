"""The entities for treasurehunt, a simple example for the purpose of a tutorial."""

# TODO: first file to write!

import numpy as np

from agentarium.primitives import Entity


class EmptyEntity(Entity):
    """An entity that represents an empty space in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
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
            env.add(self.location, Gem(env.gem_value))


class Wall(Entity):
    """An entity that represents a wall in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/wall.png"
        )


class Gem(Entity):
    """An entity that represents a gem in the treasurehunt environment."""

    def __init__(self, gem_value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/gem.png"
        )
