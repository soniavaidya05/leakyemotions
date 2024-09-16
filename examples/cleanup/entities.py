import random
from gem.primitives import Object, GridworldEnv

# --------------------------------------------------- #
# region: Environment object classes for Cleanup Task #
# --------------------------------------------------- #

class EmptyObject(Object):
  """Empty object class for the Cleanup Game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.passable = True

class Wall(Object):
  """Wall class for the Cleanup Game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg

class River(Object):
  """River class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    has_transitions = True

  def transition(self, env: GridworldEnv):
    # Add pollution with a random probability
    if random.random() > self.cfg.env.pollution_spawn_chance:
      env.spawn(self.location)

class Pollution(Object):
  """Pollution class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    has_transitions = True

  def transition(self, env: GridworldEnv):
    # Check the current tile on the beam layer for cleaning beams
    beam_location = self.location[0], self.location[1], env.beam_layer
    
    # If a cleaning beam is on this tile, spawn a river tile
    if env.world[beam_location].kind == "CleanBeam":
      env.spawn(self.location)


class AppleTree(Object):
  """Potential apple class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    has_transitions = True

  def transition(self, env: GridworldEnv):
    # If the pollution threshold has not been reached...
    if not env.pollution > self.cfg.env.pollution_threshold:
      # Add apples with a random probability
      if random.random() > self.cfg.env.apple_spawn_chance:
        env.spawn(self.location)

class Apple(Object):
  """Apple class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.value = cfg.value # Reward for eating the apple

  def transition(self, env: GridworldEnv):
    # Check the current tile on the agent layer for agents
    agent_location = self.location[0], self.location[1], env.agent_layer
    
    # If there is an agent on this tile, spawn an apple tree tile
    if env.world[agent_location].kind == "Agent":
      env.spawn(self.location)

# --------------------------------------------------- #
# endregion                                           #
# --------------------------------------------------- #


