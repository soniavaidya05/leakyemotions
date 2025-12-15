from pathlib import Path
from omegaconf import OmegaConf

from sorrel.examples.leakyemotions.main import create_run_name, resolve_config_path
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld 
from sorrel.examples.leakyemotions.env import LeakyEmotionsEnv, EmptyEntity

from sorrel.utils.logging import TensorboardLogger

if __name__ == "__main__":
  config_path = resolve_config_path('default.yaml')
  config = OmegaConf.load(config_path)

  SPAWN_PROBS = [0.] # [0.001, 0.002, 0.003]
  AGENT_VISION_RADIUS = [3]
  BUSH_MODE = ["bush"] # ["bush", "wolf", "both"]
  EMOTION_CONDITION = ["none"] # ["full", "self", "other", "none"]

  for spawn_prob in SPAWN_PROBS:
    for avr in AGENT_VISION_RADIUS:
      for mode in BUSH_MODE:
        for emotion_condition in EMOTION_CONDITION:
          print(f"=== === === === === === === === === === === === === ===")
          print(f"===   Running with the following parameter values:  ===")
          print(f"===   Bush mode: {mode:<4}                               ===")
          print(f"===   Has emotion: {emotion_condition:<10}                       ===")
          print(f"===   Agent vision radius: {avr:<1}                        ===")
          print(f"===   Bush spawn prob: {spawn_prob:<5}                        ===")
          print(f"=== === === === === === === === === === === === === ===")

          config.world.spawn_prob = spawn_prob
          config.model.agent_vision_radius = avr
          config.model.emotion_condition = emotion_condition
          config.experiment.mode = mode
          if mode == "bush":
            config.world.wolves = 0
          elif mode == "wolf":
            config.world.spawn_prob = 0
            assert config.world.wolves > 0, "Must have nonzero number of wolves in wolf mode."
          config.experiment.run_name = create_run_name(config)

          # construct the world
          world = LeakyEmotionsWorld(config=config, default_entity=EmptyEntity())
          # construct the environment
          env = LeakyEmotionsEnv(world, config) # type:ignore[arg-type]
          # run the experiment with default parameters
          name = create_run_name(config)
          env.run_experiment(logger=TensorboardLogger(
              max_epochs=config.experiment.epochs,
              log_dir=Path(__file__).parent / f'./data/tensorboard/{name}/'
          ))