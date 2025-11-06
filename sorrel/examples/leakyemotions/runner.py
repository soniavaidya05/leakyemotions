from omegaconf import OmegaConf

from sorrel.examples.leakyemotions.main import create_run_name, resolve_config_path
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld 
from sorrel.examples.leakyemotions.env import LeakyEmotionsEnv, EmptyEntity

from sorrel.utils.logging import TensorboardLogger

if __name__ == "__main__":
  config_path = resolve_config_path('default.yaml')
  config = OmegaConf.load(config_path)

  SPAWN_PROBS = [0] # [0.001, 0.002, 0.003]
  AGENT_VISION_RADIUS = [3, 4, 5]
  BUSH_MODE = ["wolf"] # ["bush", "wolf", "both"]
  HAS_EMOTION = [True, False]

  for spawn_prob in SPAWN_PROBS:
    for avr in AGENT_VISION_RADIUS:
      for mode in BUSH_MODE:
        for has_emotion in HAS_EMOTION:
          print(f"=== === === === === === === === === === === === === ===")
          print(f"===   Running with the following parameter values:  ===")
          print(f"===   Bush mode: {mode:<4}                               ===")
          print(f"===   Has emotion: {has_emotion:<5}                            ===")
          print(f"===   Agent vision radius: {avr:<1}                        ===")
          print(f"===   Bush spawn prob: {spawn_prob:<5}                        ===")
          print(f"=== === === === === === === === === === === === === ===")

          config.world.spawn_prob = spawn_prob
          config.model.agent_vision_radius = avr
          config.model.has_emotion = has_emotion
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
              log_dir=f'./data/tensorboard/{name}/'
          ))