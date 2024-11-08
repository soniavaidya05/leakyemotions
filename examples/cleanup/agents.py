import torch
from typing import Optional, Sequence

import numpy as np

from examples.trucks.agents import Memory
from gem.models.ann import ANN
from gem.primitives import Object, GridworldEnv, Location, Vector
from gem.utils import visual_field, visual_field_multilayer, one_hot_encode
from gem.models.grid_cells import positional_embedding

# ------------------- #
# region: Agent class #
# ------------------- #


class Agent(Object):
    """Cleanup agent."""

    def __init__(self, cfg, appearance, model):

        super().__init__(appearance)

        self.cfg = cfg
        self.vision = cfg.agent.agent.vision
        self.num_frames = cfg.agent.agent.num_frames
        self.direction = 2  # 90 degree rotation: default at 180 degrees (facing down)
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.has_transitions = True
        self.sprite = f"{self.cfg.root}/examples/cleanup/assets/hero.png"

        # training-related features
        self.action_type = "neural_network"
        self.model = model
        # self.episode_memory = Memory(cfg.agent.agent.memory_size)
        # self.num_memories = cfg.agent.agent.memory_size
        # self.init_rnn_state = None

        self.rotation = cfg.agent.agent.rotation

        # logging features
        self.outcome_record = {"harvest": 0, "zap": 0, "get_zapped": 0, "clean": 0}

    def sprite_loc(self) -> None:
        """Determine the agent's sprite based on the location."""
        sprite_directions = [
            f"{self.cfg.root}/examples/cleanup/assets/hero-back.png",  # up
            f"{self.cfg.root}/examples/cleanup/assets/hero-right.png",  # right
            f"{self.cfg.root}/examples/cleanup/assets/hero.png",  # down
            f"{self.cfg.root}/examples/cleanup/assets/hero-left.png",  # left
        ]
        self.sprite = sprite_directions[self.direction]

    def init_replay(self, env: GridworldEnv) -> None:
        """Fill in blank images for the LSTM."""

        state = np.zeros_like(self.current_state(env))
        action = float(len(self.action_space)+1)  # Action outside the action space
        reward = 0.0
        done = 0.0
        for _ in range(self.num_frames):
            self.model.memory.add(state, action, reward, done)

    def act(self, action: int, rotate=False) -> tuple[int, ...]:
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

            if action == 0:  # NOOP
                pass
            if action == 1:  # FORWARD
                forward_vector = Vector(1, 0, direction=self.direction)
                cur_location = Location(*self.location)
                next_location = (cur_location + forward_vector).to_tuple()
            if action == 2:  # BACK
                backward_vector = Vector(-1, 0, direction=self.direction)
                cur_location = Location(*self.location)
                next_location = (cur_location + backward_vector).to_tuple()
            if action == 3:  # TURN CLOCKWISE
                # Add 90 degrees; modulo 4 to ensure range of [0, 1, 2, 3]
                self.direction = (self.direction + 1) % 4
            if action == 4:  # TURN COUNTERCLOCKWISE
                self.direction = (self.direction - 1) % 4

            self.sprite_loc()

        else:
            if action == 0:  # UP
                self.sprite = f"{self.cfg.root}/examples/RPG/assets/hero-back.png"
                next_location = (
                    self.location[0] - 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 1:  # DOWN
                self.sprite = f"{self.cfg.root}/examples/RPG/assets/hero.png"
                next_location = (
                    self.location[0] + 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 2:  # LEFT
                self.sprite = f"{self.cfg.root}/examples/RPG/assets/hero-left.png"
                next_location = (
                    self.location[0],
                    self.location[1] - 1,
                    self.location[2],
                )
            if action == 3:  # RIGHT
                self.sprite = f"{self.cfg.root}/examples/RPG/assets/hero-right.png"
                next_location = (
                    self.location[0],
                    self.location[1] + 1,
                    self.location[2],
                )

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
            loc for loc in valid_locs if not env.world[loc.to_tuple()].kind == "Wall"
        ]

        # Then, place beams in all of the remaining valid locations.
        for loc in placeable_locs:
            if action == 5:
                env.remove(loc.to_tuple())
                env.add(
                    loc.to_tuple(), CleanBeam(self.cfg, env.appearances["CleanBeam"])
                )
            elif action == 6:
                env.remove(loc.to_tuple())
                env.add(loc.to_tuple(), ZapBeam(self.cfg, env.appearances["ZapBeam"]))

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """
        Defines the agent's observation function
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
        return positional_embedding(self.location, env, 3, 3)
    
    def current_state(self, env: GridworldEnv) -> np.ndarray:
        pov = self.pov(env)
        pos = self.embed_loc(env)
        state = np.concatenate((pov, pos))
        prev_states = self.model.memory.current_state(stacked_frames=self.num_frames-1)
        current_state = np.vstack((prev_states, state))
    
        return current_state
    
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an experience to the memory."""
        self.model.memory.add(state, action, reward, done)

    def transition(self, env: GridworldEnv, state, action):
        """Changes the world based on action taken."""
        reward = 0

        # Attempt the transition
        attempted_location = self.act(action)

        # Generate beams, if necessary
        if action in [5, 6]:
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

    def reset(self) -> None:
        self.model.memory.clear()
        # self.init_replay()


# ------------------- #
# endregion           #
# ------------------- #

# ------------------- #
# region: Beams       #
# ------------------- #


class Beam(Object):
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


# ------------------- #
# endregion           #
# ------------------- #

"""
-------------------
old functions below
-------------------
"""


def pov_old(self, env) -> torch.Tensor:
    """
    Defines the agent's observation function
    """
    # Get the previous state
    previous_state = self.episode_memory.get_last_memory("states")

    # Get the frames from the previous state
    current_state = previous_state.clone()

    current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

    # If the environment is a full MDP, get the whole world image
    if env.full_mdp:
        image = visual_field_multilayer(env.world, env.color_map, channels=env.channels)
    # Otherwise, use the agent observation function
    else:
        image = visual_field_multilayer(
            env.world, env.color_map, self.location, self.vision, env.channels
        )

    # Update the latest state to the observation
    state_now = torch.tensor(image).unsqueeze(0)
    current_state[:, -1, :, :, :] = state_now

    return current_state


def init_replay(self) -> None:
    """Fill in blank images for the LSTM."""

    priority = torch.tensor(0.1)
    num_frames = self.model.num_frames
    if self.cfg.env.full_mdp:
        state = torch.zeros(1, num_frames, *self.model.state_size).float()
    else:
        # Number of one-hot code channels
        C = len(self.appearance)
        H = W = self.vision * 2 + 1
        state = torch.zeros(1, num_frames, C, H, W).float()

    action = torch.tensor(7.0)  # Action outside the action space
    reward = torch.tensor(0.0)
    done = torch.tensor(0.0)
    exp = (priority, (state, action, reward, state, done))
    self.episode_memory.append(exp)

def pov_stack(self, env: GridworldEnv) -> torch.Tensor:
    """
    Defines the agent's observation function
    """
    # Get the previous state
    previous_state = self.episode_memory.get_last_memory("states")

    # Get the frames from the previous state
    current_state = previous_state.clone()

    current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

    # If the environment is a full MDP, get the whole world image
    if env.full_mdp:
        image = visual_field_multilayer(env.world, env.color_map, channels=env.channels)
    # Otherwise, use the agent observation function
    else:
        image = visual_field_multilayer(
            env.world, env.color_map, self.location, self.vision, env.channels
        )

    # Update the latest state to the observation
    state_now = torch.tensor(image).unsqueeze(0)
    current_state[:, -1, :, :, :] = state_now

    return current_state