# ------------------------ #
# region: Imports          #
# general imports
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch

# for configs
from omegaconf import DictConfig, OmegaConf

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.entities import Entity
from sorrel.examples.cleanup.agents import CleanupAgent, CleanupObservation
from sorrel.examples.cleanup.env import Cleanup
from sorrel.models.pytorch import PyTorchIQN
from sorrel.utils.logging import ConsoleLogger
from sorrel.utils.visualization import (
    ImageRenderer,
    animate_gif,
    image_from_array,
    render_sprite,
)

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


def setup(cfg, **kwargs) -> Cleanup:
    """Set up the environment and everything within it."""

    agents = []
    # make the agents
    for _ in range(cfg.agent.agent.num):

        agent_vision_radius = cfg.agent.agent.obs.vision
        observation_spec = CleanupObservation(
            entity_list=ENTITY_LIST, vision_radius=agent_vision_radius
        )
        action_spec = ActionSpec(["up", "down", "left", "right", "clean", "zap"])

        model = PyTorchIQN(
            input_size=observation_spec.input_size,
            action_space=action_spec.n_actions,
            seed=torch.random.seed(),
            n_frames=cfg.agent.agent.obs.n_frames,
            **cfg.model.iqn.parameters,
        )

        if "load_weights" in kwargs:
            path = kwargs.get("load_weights")
            if isinstance(path, str | os.PathLike):
                model.load(file_path=path)

        agents.append(
            CleanupAgent(
                observation_spec=observation_spec, action_spec=action_spec, model=model
            )
        )

    env = Cleanup(cfg, agents)
    return env


def run(env: Cleanup, **kwargs):
    """Run the experiment."""
    cfg = env.cfg

    logger = ConsoleLogger(cfg.experiment.epochs)
    renderer = ImageRenderer(
        cfg.experiment.name, cfg.experiment.record_period, cfg.experiment.max_turns
    )

    for epoch in range(cfg.experiment.epochs + 1):

        total_loss = 0

        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent: CleanupAgent
            agent.model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            renderer.add_image(env, epoch)
            env.take_turn()

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in env.agents:
            loss = agent.model.train_step()
            total_loss += loss

        # Record the turn
        epsilon = env.agents[0].model.epsilon
        logger.record_turn(epoch, total_loss, env.game_score, epsilon)

        # Save a gif when
        renderer.save_gif(epoch, folder=Path(__file__).parent / "./data/")

        # update epsilon
        for agent in env.agents:
            agent.model.epsilon_decay(cfg.experiment.epsilon_decay)

    if "save_weights" in kwargs:
        for i, agent in enumerate(env.agents):
            file_path = os.path.abspath(
                f'./checkpoints/{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}_{i}.pkl'
            )
            agent.model.save(file_path=file_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-weights", "-w", type=str, help="Path to pretrained model.", default=None
    )
    parser.add_argument(
        "--logger",
        "-l",
        help="Logger type",
        default="console",
        choices=["console", "tensorboard"],
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    env = setup(
        cfg,
        # load_weights=args.load_weights,
    )
    run(env, save_weights=True)


if __name__ == "__main__":
    main()
