from gem.primitives import Object, GridworldEnv

class EmptyObject(Object):
  """Empty object class for the Cleanup Game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.passable = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/empty.png'

class Water(Object):
  """River class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.sprite = f'{cfg.root}/examples/cleanup/assets/water.png'

class Land(Object):
  """Potential apple class for the Cleanup game."""
  def __init__(self, cfg, appearance):
    super().__init__(appearance)
    self.cfg = cfg
    self.has_transitions = True
    self.sprite = f'{cfg.root}/examples/cleanup/assets/grass.png'

