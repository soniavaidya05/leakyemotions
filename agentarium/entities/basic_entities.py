from agentarium.entities.entity import Entity

class Wall(Entity):
    """A basic entity that represents a wall."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact

class EmptyEntity(Entity):
    """A basic entity that represents a passable empty space."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces

class Gem(Entity):
    """An entity that represents a rewarding object in an environment."""

    def __init__(self, gem_value: float | int):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value