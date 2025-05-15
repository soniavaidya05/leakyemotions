"""The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports

# Import primitive types
from sorrel.environments import GridworldEnv

# end imports


# begin treasurehunt
class Treasurehunt(GridworldEnv):
    """Treasurehunt environment."""

    def __init__(self, height, width, default_entity, gem_value, spawn_prob, max_turns):
        layers = 2
        super().__init__(height, width, layers, default_entity)

        self.gem_value = gem_value
        self.spawn_prob = spawn_prob
        self.max_turns = max_turns


# end treasurehunt
