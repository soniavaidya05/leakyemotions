from agentarium.primitives import Entity, GridworldEnv

import random

class EmptyEntity(Entity):
  """Empty entity class for the AI Economist game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.passable = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/empty.png'

class Water(Entity):
  """Impassable area for the AI Economist game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.sprite = f'{cfg.root}/examples/cleanup/assets/water.png'

class Land(Entity):
  """Passable area for the AI Economist game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.sprite = f'{cfg.root}/examples/cleanup/assets/grass.png'

class WoodNode(Entity):
  """Potential wood area for the AI Economist game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self._sprite = f'{self.cfg.root}/examples/cleanup/assets/grass.png'
    self.has_transitions = True
    self.resources = 0
    if random.random() < self.cfg.env.resource_spawn_chance:
      self.resources = random.randint(5, 10)
      self.sprite('apple_grass')

  @property
  def sprite(self) -> str:
    return self._sprite
  
  @sprite.setter
  def sprite(self, new_sprite: str):
    self._sprite = f'{self.cfg.root}/examples/cleanup/assets/' + new_sprite + '.png'

  def transition(self, env: GridworldEnv) -> None:
    if self.resources == 0:
      self.sprite('grass')
      if random.random() < self.cfg.env.resource_spawn_chance:
        self.resources = random.randint(5, 10)
        self.sprite('apple_grass')



      


