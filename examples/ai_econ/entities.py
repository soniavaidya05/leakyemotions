import numpy as np

from agentarium.entities import Entity
from agentarium.environments import GridworldEnv


# Entities on multiple layers
class Wall(Entity):
    """Impassable walls for the AI Economist game."""

    def __init__(self):
        super().__init__()
        self.sprite = f"./assets/wall.png"


class EmptyEntity(Entity):
    """Empty entity class for the AI Economist game."""

    def __init__(self):
        super().__init__()
        self.passable = True
        self.sprite = f"./assets/empty.png"


# Entities on layer 0 (bottom layer)


class Land(Entity):
    """Empty land (no resources) for the AI Economist game."""

    def __init__(self):
        super().__init__()
        self.sprite = f"./assets/grass.png"


# Entities on layer 1 (middle layer)


class WoodNode(Entity):
    """Potential wood area for the AI Economist game."""

    def __init__(self, renew_chance: float, renew_amount: int):
        super().__init__()
        self.sprite = f"./assets/grass.png"
        self.passable = True
        self.has_transitions = True
        self.renew_chance = renew_chance
        self.renew_amount = renew_amount
        self.num_resources = 0
        self.renew()

    def renew(self) -> None:
        """Sets num_resources at this node to renew_amount with chance of renew_chance."""
        if np.random.random() < self.renew_chance:
            self.num_resources = self.renew_amount
            self.sprite = f"./assets/wood.png"

    def transition(self, env: GridworldEnv) -> None:
        """If no resources are left, update the sprite; then, attempt to renew the node."""
        if self.num_resources == 0:
            self.sprite = f"./assets/wood.png"  # NOTE: change the sprite when agents deplete this instead?
            self.renew()


class StoneNode(Entity):
    """Potential stone area for the AI Economist game."""

    def __init__(self, renew_chance: float, renew_amount: int):
        super().__init__()
        self.sprite = f"./assets/grass.png"
        self.passable = True
        self.has_transitions = True
        self.renew_chance = renew_chance
        self.renew_amount = renew_amount
        self.num_resources = 0
        self.renew()

    def renew(self) -> None:
        """Sets num_resources at this node to renew_amount with chance of renew_chance."""
        if np.random.random() < self.renew_chance:
            self.num_resources = self.renew_amount
            self.sprite = f"./assets/stone.png"

    def transition(self, env: GridworldEnv) -> None:
        """If no resources are left, update the sprite; then, attempt to renew the node."""
        if self.num_resources == 0:
            self.sprite = f"./assets/stone.png"  # NOTE: change the sprite when agents deplete this instead?
            self.renew()


# Entities on layer 3 (top layer)


class BuyerWoodSignal(Entity):
    """A signal that can be used by Markets to signal interest in buying wood."""

    def __init__(self):
        super().__init__()
        # doesn't have a sprite yet


class BuyerStoneSignal(Entity):
    """A signal that can be used by Markets to signal interest in buying stone."""

    def __init__(self):
        super().__init__()
        # doesn't have a sprite yet


class SellerWoodSignal(Entity):
    """A signal that can be used by Agents to signal interest in selling wood."""

    def __init__(self):
        super().__init__()
        # doesn't have a sprite yet


class SellerStoneSignal(Entity):
    """A signal that can be used by Agents to signal interest in selling stone."""

    def __init__(self):
        super().__init__()
        # doesn't have a sprite yet
