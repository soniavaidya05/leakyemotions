# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

# Import base packages
import numpy as np

# Import sorrel-specific packages
from sorrel.entities import Entity
from sorrel.environments import GridworldEnv

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #


class Cleanup(GridworldEnv):
    """Cleanup Environment."""

    def __init__(
        self,
        cfg,
        default_entity: Entity
    ):
        self.cfg = cfg
        self.channels = (
            cfg.agent.agent.obs.channels
        )  # default: # of entity classes + 1 (agent class) + 2 (beam types)
        self.full_mdp = cfg.env.full_mdp
        self.object_layer = 0
        self.agent_layer = 1
        self.beam_layer = 2
        self.pollution = 0
        super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, default_entity)
        self.mode = cfg.env.mode
        self.max_turns = cfg.experiment.max_turns
        self.pollution_threshold = cfg.env.pollution_threshold
        self.pollution_spawn_chance = cfg.env.pollution_spawn_chance
        self.apple_spawn_chance = cfg.env.apple_spawn_chance
        self.initial_apples = cfg.env.initial_apples
        self.turn = 0

    def measure_pollution(self) -> float:
        pollution_tiles = 0
        river_tiles = 0
        for index, x in np.ndenumerate(self.world):
            x: Entity
            if x.kind == "Pollution":
                pollution_tiles += 1
                river_tiles += 1
            elif x.kind == "River":
                river_tiles += 1
        return pollution_tiles / river_tiles
