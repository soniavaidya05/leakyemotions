import numpy as np

from agentarium.embedding import positional_embedding
from agentarium.location import Location, Vector
from agentarium.primitives import Agent, Entity, GridworldEnv
from agentarium.utils import one_hot_encode
from agentarium.visual_field import visual_field_multilayer

# --------------------------- #
# region: Cleanup agent class #
# --------------------------- #


class CleanupAgent(Agent):
    """Cleanup agent."""

    def __init__(self, cfg, model):

        # Instantiate basic agent
        appearance = color_map(cfg.obs.channels)
        action_space = [0, 1, 2, 3, 4, 5, 6]
        super().__init__(cfg, appearance, model, action_space)

        # Additional attributes
        self.num_frames = cfg.obs.num_frames
        self.embedding_size = cfg.obs.embeddings
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.rotation = cfg.agent.agent.rotation
        self.sprite_path = f"{cfg.root}/examples/cleanup/assets/"
        self._sprite = self.sprite_path + "hero" + ".png"

        # logging features
        self.outcome_record = {"harvest": 0, "zap": 0, "get_zapped": 0, "clean": 0}

    @property
    def sprite(self):
        """Agent sprite."""
        return self._sprite

    @sprite.setter
    def sprite(self, new_sprite):
        """Update the agent sprite with the name of a new sprite."""
        self._sprite = self.sprite_path + new_sprite + ".png"

    def sprite_loc(self) -> None:
        """Determine the agent's sprite based on the location."""
        sprite_directions = [
            "hero-back",  # up
            "hero-right",  # right
            "hero",  # down
            "hero-left",  # left
        ]
        self.sprite(sprite_directions[self.direction])

    def init_replay(self, env: GridworldEnv) -> None:
        """Fill in blank images for the memory buffer."""

        state = np.zeros_like(self.current_state(env))
        action = 0  # Action outside the action space
        reward = 0.0
        done = 0.0
        for _ in range(self.num_frames):
            self.model.memory.add(state, action, reward, done)

    def act(self, action: int) -> tuple[int, ...]:
        """Act on the environment.

        Params:
            action: (int) An integer indicating the action to take.

        Return:
            tuple[int, ...] A location tuple indicating the updated
            location of the agent.
        """

        # Default location
        next_location = self.location

        if self.rotation:
            if action == 0:  # FORWARD
                forward_vector = Vector(1, 0, direction=self.direction)
                cur_location = Location(*self.location)
                next_location = (cur_location + forward_vector).to_tuple()
            if action == 1:  # BACK
                backward_vector = Vector(-1, 0, direction=self.direction)
                cur_location = Location(*self.location)
                next_location = (cur_location + backward_vector).to_tuple()
            if action == 2:  # TURN CLOCKWISE
                # Add 90 degrees; modulo 4 to ensure range of [0, 1, 2, 3]
                self.direction = (self.direction + 1) % 4
            if action == 3:  # TURN COUNTERCLOCKWISE
                self.direction = (self.direction - 1) % 4

        else:
            if action == 0:  # UP
                self.direction = 0
                next_location = (
                    self.location[0] - 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 1:  # DOWN
                self.direction = 2
                next_location = (
                    self.location[0] + 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 2:  # LEFT
                self.direction = 3
                next_location = (
                    self.location[0],
                    self.location[1] - 1,
                    self.location[2],
                )
            if action == 3:  # RIGHT
                self.direction = 1
                next_location = (
                    self.location[0],
                    self.location[1] + 1,
                    self.location[2],
                )

        self.sprite_loc()

        return next_location

    def spawn_beam(self, env: GridworldEnv, action):
        """Generate a beam extending cfg.agent.agent.beam_radius pixels
        out in front of the agent."""

        # Get the tiles above and adjacent to the agent.
        up_vector = Vector(0, 0, layer=1, direction=self.direction)
        forward_vector = Vector(1, 0, direction=self.direction)
        right_vector = Vector(0, 1, direction=self.direction)
        left_vector = Vector(0, -1, direction=self.direction)

        tile_above = Location(*self.location) + up_vector

        # Candidate beam locations:
        #   1. (1, i+1) tiles ahead of the tile above the agent
        #   2. (0, i) tiles ahead of the tile above and to the right/left of the agent.
        beam_locs = (
            [
                (tile_above + (forward_vector * i))
                for i in range(1, self.cfg.agent.agent.beam_radius + 1)
            ]
            + [
                (tile_above + (right_vector) + (forward_vector * i))
                for i in range(self.cfg.agent.agent.beam_radius)
            ]
            + [
                (tile_above + (left_vector) + (forward_vector * i))
                for i in range(self.cfg.agent.agent.beam_radius)
            ]
        )

        # Check beam layer to determine which locations are valid...
        valid_locs = [loc for loc in beam_locs if env.valid_location(loc)]

        # Exclude any locations that have walls...
        placeable_locs = [
            loc for loc in valid_locs if not str(env.observe(loc.to_tuple())) == "Wall"
        ]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            if action == 4:
                env.remove(loc.to_tuple())
                env.add(
                    loc.to_tuple(), CleanBeam(self.cfg, env.appearances["CleanBeam"])
                )
            elif action == 5:
                env.remove(loc.to_tuple())
                env.add(loc.to_tuple(), ZapBeam(self.cfg, env.appearances["ZapBeam"]))

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """
        Defines the agent's visual field function.

        Parameters:
            env: (GridworldEnv) The environment to observe.

        Return:
            np.ndarray: The visual field of the agent.
        """

        # If the environment is a full MDP, get the whole world image
        if env.full_mdp:
            image = visual_field_multilayer(
                env.world, env.color_map, channels=env.channels
            )
        # Otherwise, use the agent observation function
        else:
            image = visual_field_multilayer(
                env.world, env.color_map, self.location, self.vision, env.channels
            )

        current_state = image.flatten()

        return current_state

    def embed_loc(self, env: GridworldEnv) -> np.ndarray:
        """
        Obtain the agent's positional embedding.

        Parameters:
            env: (GridworldEnv) The environment to observe.

        Return:
            np.ndarray: The agent's positional code.
        """
        return positional_embedding(
            self.location, env, self.embedding_size, self.embedding_size
        )

    def current_state(self, env: GridworldEnv) -> np.ndarray:
        """
        Obtain the agent's observation function.

        Parameters:
            env: (GridworldEnv) The environment to observe.

        Return:
            np.ndarray: The agent's positional code.
        """
        pov = self.pov(env)
        pos = self.embed_loc(env)
        ohe = one_hot_encode(self.direction, 4)
        state = np.concatenate((pov, pos, ohe))
        prev_states = self.model.memory.current_state(
            stacked_frames=self.num_frames - 1
        )
        current_state = np.vstack((prev_states, state))

        return current_state

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the memory.

        Parameters:
            state: (np.ndarray)
            action: (int)
            reward: (float)
            done: (bool)
        """
        self.model.memory.add(state, action, reward, done)

    def transition(self, env: GridworldEnv, action: int):
        """Changes the world based on action taken."""
        reward = 0

        # Attempt the transition
        attempted_location = self.act(action)

        # Generate beams, if necessary
        if action in [4, 5]:
            self.spawn_beam(env, action)

        # Get the candidate reward objects
        reward_locations = [
            (attempted_location[0], attempted_location[1], i)
            for i in range(env.world.shape[2])
        ]
        reward_objects = [env.observe(loc) for loc in reward_locations]

        # Complete the transition
        env.move(self, attempted_location)

        # Get the interaction reward
        for obj in reward_objects:
            reward += obj.value

        # Get the next state
        location_code = positional_embedding(self.location, env, 3, 3)
        direction = one_hot_encode(self.direction, 4)
        # next_state = np.concatenate(
        #     [self.pov(env).flatten(), location_code, direction]
        # ).reshape(1, -1)
        next_state = self.pov_stack(env)

        return reward, next_state, False

    def reset(self, env: GridworldEnv) -> None:
        # self.model.memory.clear()
        self.init_replay(env)


# --------------------------- #
# endregion                   #
# --------------------------- #

# --------------------------- #
# region: Beams               #
# --------------------------- #


class Beam(Entity):
    """Generic beam class for agent beams."""

    def __init__(self, cfg, appearance):
        super().__init__(appearance)
        self.cfg = cfg
        self.sprite = f"{cfg.root}/examples/cleanup/assets/beam.png"
        self.turn_counter = 0

    def transition(self, env: GridworldEnv):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            env.spawn(self.location)
        else:
            self.turn_counter += 1


class CleanBeam(Beam):
    def __init__(self, cfg, appearance):
        super().__init__(cfg, appearance)


class ZapBeam(Beam):
    def __init__(self, cfg, appearance):
        super().__init__(cfg, appearance)
        self.sprite = f"{cfg.root}/examples/cleanup/assets/zap.png"
        self.value = -1


# --------------------------- #
# endregion                   #
# --------------------------- #

# --------------------------- #
# region: Color map           #
# --------------------------- #


def color_map(self, C: int) -> dict:
    """Color map for visualization."""
    assert C in [3, 8], "Must use 3 [RGB] or 8 channels."
    if C == 8:
        colors = {
            "EmptyEntity": [0 for _ in range(self.channels)],
            "Agent": [255 if x == 0 else 0 for x in range(self.channels)],
            "Wall": [255 if x == 1 else 0 for x in range(self.channels)],
            "Apple": [255 if x == 2 else 0 for x in range(self.channels)],
            "AppleTree": [255 if x == 3 else 0 for x in range(self.channels)],
            "River": [255 if x == 4 else 0 for x in range(self.channels)],
            "Pollution": [255 if x == 5 else 0 for x in range(self.channels)],
            "CleanBeam": [255 if x == 6 else 0 for x in range(self.channels)],
            "ZapBeam": [255 if x == 7 else 0 for x in range(self.channels)],
        }
    else:
        colors = {
            "EmptyEntity": [0.0, 0.0, 0.0],
            "Agent": [150.0, 150.0, 150.0],
            "Wall": [50.0, 50.0, 50.0],
            "Apple": [0.0, 200.0, 0.0],
            "AppleTree": [100.0, 100.0, 0.0],
            "River": [0.0, 0.0, 200.0],
            "Pollution": [0, 100.0, 200.0],
            "CleanBeam": [200.0, 255.0, 200.0],
            "ZapBeam": [255.0, 200.0, 200.0],
        }
    return colors


# --------------------------- #
# endregion                   #
# --------------------------- #
