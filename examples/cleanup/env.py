# --------------------------------- #
# region: Imports                   #
# --------------------------------- #

# Import base packages
import numpy as np

# Import agentarium-specific packages
from agentarium.agents import Agent
from agentarium.entities import Entity
from agentarium.environments import GridworldEnv
from examples.cleanup.entities import (
    EmptyEntity,
    Sand,
    Wall,
    River,
    Pollution,
    AppleTree,
    Apple
)
from examples.cleanup.agents import CleanupAgent
from agentarium.config import Cfg

# --------------------------------- #
# endregion: Imports                #
# --------------------------------- #

class Cleanup(GridworldEnv):
  """Cleanup Environment."""

  def __init__(
    self,
    cfg: Cfg,
    agents: list[CleanupAgent],
    mode: str = "DEFAULT"
  ):
    self.cfg = cfg
    self.channels = cfg.agent.agent.obs.channels # default: # of entity classes + 1 (agent class) + 2 (beam types)
    self.full_mdp = cfg.env.full_mdp
    self.agents = agents
    self.object_layer = 0
    self.agent_layer = 1
    self.beam_layer = 2
    self.pollution = 0
    super().__init__(cfg.env.height, cfg.env.width, cfg.env.layers, eval(cfg.env.default_object)(cfg))
    self.mode = mode
    self.turn = 0
    self.populate()
  

  def populate(self) -> None:
  
    spawn_points = []
    apple_spawn_points = []
        
    # First, create the walls
    for index in np.ndindex(self.world.shape):
      H, W, L = index

      # If the index is the first or last, replace the location with a wall
      if H in [0, self.height - 1] or W in [0, self.width - 1]:
        self.add(index, Wall(self.cfg))
      # Define river, orchard, and potential agent spawn points
      elif L == 0:
        if self.mode != "APPLE":
          # Top third = river
          if H > 0 and H < (self.height // 3):
            self.add(index, River(self.cfg))
          # Bottom third = orchard
          elif H > (self.height - 1 - (self.height // 3)) and H < (self.height - 1):
            self.add(index, AppleTree(self.cfg))
            apple_spawn_points.append(index)
          # Middle third = potential agent spawn points
          else:
            self.add(index, Sand(self.cfg))
            spawn_index = [index[0], index[1], self.agent_layer]
            spawn_points.append(spawn_index)
        else:
          self.add(index, AppleTree(self.cfg))
          if ((H % 2) == 0) and ((W % 2) == 0):
            spawn_index = [index[0], index[1], self.agent_layer]
            spawn_points.append(spawn_index)
          else:
            apple_spawn_points.append(index)

    # Place apples randomly based on the spawn points chosen
    loc_index = np.random.choice(len(apple_spawn_points), size = self.cfg.env.initial_apples, replace = False)
    locs = [apple_spawn_points[i] for i in loc_index]
    for loc in locs:
      loc = tuple(loc)
      self.add(loc, Apple(self.cfg))
      
    # Place agents randomly based on the spawn points chosen
    loc_index = np.random.choice(len(spawn_points), size = len(self.agents), replace = False)
    locs = [spawn_points[i] for i in loc_index]
    for loc, agent in zip(locs, self.agents):
      loc = tuple(loc)
      self.add(loc, agent)


  def get_entities_for_transition(self) -> list[Entity]:
    entities = []
    for index, x in np.ndenumerate(self.world):
      if not isinstance(x, Wall) and not isinstance(x, Agent):
        entities.append(x)
    return entities
  

  def measure_pollution(self) -> float:
    pollution_tiles = 0
    river_tiles = 0
    for index, x in np.ndenumerate(self.world):
      if x.kind == "Pollution":
        pollution_tiles += 1
        river_tiles += 1
      elif x.kind == "River":
        river_tiles += 1
    return pollution_tiles / river_tiles


  def take_turn(self) -> None:
    """
    Environment transition function.
    """

    self.turn += 1
    self.pollution = self.measure_pollution()

    for entity in self.get_entities_for_transition():
      entity.transition(self)

    for agent in self.agents:
      state, action, reward, done = agent.transition(self)
      agent.add_memory(state, action, reward, done)

      if self.turn >= self.cfg.experiment.max_turns or done:
        agent.add_final_memory(self) 


  def reset(self):
    """Reset the environment."""
    self.create_world()
    self.populate()
    for agent in self.agents:
      agent.reset(self)
      agent.init_replay(self)




    

      



      
    


  
