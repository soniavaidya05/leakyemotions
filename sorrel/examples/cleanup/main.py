# ------------------------ #
# region: Imports          #
# general imports
import os
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch

# for configs
from omegaconf import DictConfig

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.examples.cleanup.agents import CleanupAgent, CleanupObservation
from sorrel.examples.cleanup.entities import (
    EmptyEntity,
    Wall,
    River,
    Pollution,
    AppleTree,
    Apple,
    Sand
)
from sorrel.examples.cleanup.env import Cleanup
from sorrel.experiment import Experiment
from sorrel.models.pytorch import PyTorchIQN

# endregion                #
# ------------------------ #

# Experiment parameters
ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "River",
    "Pollution",
    "AppleTree",
    "Apple",
    "CleanBeam",
    "ZapBeam",
    "CleanupAgent",
]

class CleanupExperiment(Experiment[Cleanup]):

    def setup_agents(self):
        """Set up the agents."""
        agents = []
        for _ in range(self.config.agent.agent.num):
            agent_vision_radius = self.config.agent.agent.obs.vision
            observation_spec = CleanupObservation(
                entity_list=ENTITY_LIST, vision_radius=agent_vision_radius
            )
            action_spec = ActionSpec(["up", "down", "left", "right", "clean", "zap"])

            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                seed=torch.random.seed(),
                n_frames=self.config.agent.agent.obs.n_frames,
                **self.config.model.iqn.parameters,
            )

            # if "load_weights" in kwargs:
            #     path = kwargs.get("load_weights")
            #     if isinstance(path, str | os.PathLike):
            #         model.load(file_path=path)

            agents.append(
                CleanupAgent(
                    observation_spec=observation_spec, action_spec=action_spec, model=model
                )
            ) 

        self.agents = agents

    def populate_environment(self):
        spawn_points = []
        apple_spawn_points = []

        # First, create the walls
        for index in np.ndindex(self.env.world.shape):
            H, W, L = index

            # If the index is the first or last, replace the location with a wall
            if H in [0, self.env.height - 1] or W in [0, self.env.width - 1]:
                self.env.add(index, Wall())
            # Define river, orchard, and potential agent spawn points
            elif L == 0:
                if self.env.mode != "APPLE":
                    # Top third = river (plus section that extends further down)
                    if (H > 0 and H < (self.env.height // 3)) or (
                        H < ((self.env.height // 3) * 2 - 1)
                        and W in [self.env.width // 3, 1 + self.env.width // 3]
                    ):
                        self.env.add(index, River())
                    # Bottom third = orchard
                    elif H > (self.env.height - 1 - (self.env.height // 3)) and H < (
                        self.env.height - 1
                    ):
                        self.env.add(index, AppleTree())
                        apple_spawn_points.append(index)
                    # Middle third = potential agent spawn points
                    else:
                        self.env.add(index, Sand())
                        spawn_index = [index[0], index[1], self.env.agent_layer]
                        spawn_points.append(spawn_index)
                else:
                    self.env.add(index, AppleTree())
                    if ((H % 2) == 0) and ((W % 2) == 0):
                        spawn_index = [index[0], index[1], self.env.agent_layer]
                        spawn_points.append(spawn_index)
                    else:
                        apple_spawn_points.append(index)

        # Place apples randomly based on the spawn points chosen
        loc_index = np.random.choice(
            len(apple_spawn_points), size=self.env.initial_apples, replace=False
        )
        locs = [apple_spawn_points[i] for i in loc_index]
        for loc in locs:
            loc = tuple(loc)
            self.env.add(loc, Apple())

        # Place agents randomly based on the spawn points chosen
        loc_index = np.random.choice(
            len(spawn_points), size=len(self.agents), replace=False
        )
        locs = [spawn_points[i] for i in loc_index]
        for loc, agent in zip(locs, self.agents):
            loc = tuple(loc)
            self.env.add(loc, agent)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--load-weights", "-l", type=str, help="Path to pretrained model.", default=None
    # )
    # parser.add_argument(
    #     "--save-weights", "-s", type=str, help="Path to pretrained model.", default=None
    # )
    # parser.add_argument(
    #     "--logger",
    #     "-g",
    #     help="Logger type",
    #     default="console",
    #     choices=["console", "tensorboard"],
    # )
    # parser.add_argument("--verbose", "-v", action="count", default=0)
    # args = parser.parse_args()
    
    env = Cleanup(cfg=cfg, default_entity=EmptyEntity())
    experiment = CleanupExperiment(env, cfg)
    experiment.run()

# begin main
if __name__ == "__main__":
    main()