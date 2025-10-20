from omegaconf import OmegaConf
from datetime import datetime
import os
import argparse
from pathlib import Path

from sorrel.examples.leakyemotions.entities import EmptyEntity
from sorrel.examples.leakyemotions.env import LeakyEmotionsEnv
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld

from sorrel.utils.logging import TensorboardLogger


def create_run_name(config):
    """
    Create a descriptive name for this run to display in TensorBoard.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    world_cfg = config.world
    run_name = (
        f"{config.experiment.mode}-mode_"
        f"{config.model.has_emotion}-emotion_"
        f"{config.model.agent_vision_radius}-viz_"
        f"{world_cfg.agents}-agent_"
        f"{world_cfg.wolves}-wolf_"
        f"{world_cfg.spawn_prob}-spawn_"
        # f"{world_cfg.width}x{world_cfg.height}_" # Static for now
        f"{timestamp}"
    )
    
    return run_name



def resolve_config_path(config_name):
    """
    Resolve config name to full path. Automatically looks in the configs folder.
    
    Args:
        config_name: Either a filename like "default.yaml" or full path
    
    Returns:
        Path object to the config file
    """
    config_path = Path(config_name)
    
    # If it's already a full path that exists, use it
    if config_path.exists():
        return config_path
    
    # Otherwise, look in leaky emotion's configs directory
    configs_dir = Path(__file__).parent / "configs"
    potential_path = configs_dir / config_name
    
    if potential_path.exists():
        return potential_path
    
    # Not found in either location
    raise FileNotFoundError(
        f"Config file not found: {config_name}\n"
        f"Searched in:\n"
        f"  - {config_path.absolute()}\n"
        f"  - {potential_path.absolute()}"
    )


# begin main
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["bush", "wolf", "bush_wolf"], required=True)
    args = parser.parse_args()

    
    config_path = resolve_config_path(args.config)
    
    config = OmegaConf.load(config_path)

    if args.mode == "bush":
        config.world.wolves = 0
        print("Mode: BUSH (no wolves)")
    elif args.mode == "wolf":
        config.world.spawn_prob = 0
        print("Mode: WOLF (no bushes)")
    elif args.mode == "bush_wolf":
        print("Mode: BUSH_WOLF (both wolves and bushes)")

    # exp_dir = create_dir("./data/tensorboard", args.mode, config)

    # construct the world
    env = LeakyEmotionsWorld(config=config, default_entity=EmptyEntity())
    # construct the environment
    experiment = LeakyEmotionsEnv(env, config) # type:ignore[arg-type]
    # run the experiment with default parameters
    name = create_run_name(config)
    experiment.run_experiment(logger=TensorboardLogger(
        max_epochs=config.experiment.epochs,
        log_dir=f'./data/tensorboard/{name}/'
    ))

# end main
