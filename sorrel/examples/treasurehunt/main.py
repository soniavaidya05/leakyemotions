# begin imports
# general imports
from pathlib import Path

import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec

# imports from our example
from sorrel.examples.treasurehunt.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt.env import Treasurehunt
from sorrel.experiment import Experiment

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import ConsoleLogger
from sorrel.utils.visualization import (
    ImageRenderer,
    animate,
    image_from_array,
    render_sprite,
)

# end imports

# begin parameters
EPOCHS = 500
MAX_TURNS = 100
EPSILON_DECAY = 0.0001
ENTITY_LIST = ["EmptyEntity", "Wall", "Sand", "Gem", "TreasurehuntAgent"]
RECORD_PERIOD = 50  # how many epochs in each data recording period
# end parameters


class TreasurehuntExperiment(Experiment[Treasurehunt]):
    """The experiment for treasurehunt."""

    def __init__(self, env: Treasurehunt, config: dict) -> None:
        super().__init__(env, config)

    def setup_agents(self):
        """Set up the agents."""
        agent_num = 2
        agents = []
        for _ in range(agent_num):
            observation_spec = OneHotObservationSpec(
                ENTITY_LIST,
                full_view=False,
                vision_radius=self.config.agent.agent_vision_radius,
            )
            observation_spec.override_input_size(
                np.array(observation_spec.input_size).reshape(1, -1).tolist()
            )
            action_spec = ActionSpec(["up", "down", "left", "right"])

            model = PyTorchIQN(
                # the agent can see r blocks on each side, so the size of the observation is (2r+1) * (2r+1)
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.7,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )

            agents.append(
                TreasurehuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the treasurehunt world by creating walls, then randomly spawning the
        agents.

        Note that every space is already filled with EmptyEntity as part of
        super().__init__().
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.env.world.shape):
            y, x, z = index
            if y in [0, self.env.height - 1] or x in [0, self.env.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.env.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.env.add(index, Sand())
            elif (
                z == 1
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.env.add(loc, agent)


# begin main
if __name__ == "__main__":

    # object configurations
    configs = {
        "experiment": {
            "epochs": EPOCHS,
            "max_turns": MAX_TURNS,
            "record_period": RECORD_PERIOD,
        },
        "agent": {
            "epsilon_decay": EPSILON_DECAY,
        },
        "model": {
            "agent_vision_radius": 2,
        },
    }
    world_height = 10
    world_width = 10
    gem_value = 10
    spawn_prob = 0.002
    agent_vision_radius = 2

    # make the environment
    env = Treasurehunt(
        world_height, world_width, EmptyEntity, gem_value, spawn_prob, MAX_TURNS
    )
    experiment = TreasurehuntExperiment(env, configs)
    experiment.run()

# end main
