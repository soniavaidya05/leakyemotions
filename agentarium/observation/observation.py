import numpy as np
from agentarium.observation.visual_field import visual_field
from agentarium.utils.helpers import one_hot_encode
from agentarium.primitives import GridworldEnv

class ObservationSpec:
  """Observation specification class for Agentarium agents."""
  def __init__(
      self,
      entity_list: list[str],
      vision_radius: int | None = None
  ):
    if not isinstance(vision_radius, int):
      self.vision_radius = None
    else:
      self.vision_radius = vision_radius
    self.entity_map = self.generate_map(entity_list)

  def generate_map(self, entity_list: list[str]) -> dict[str, list[float]]:
    r"""Given a list of entities, return a dictionary of appearance values
    for the agent's observation function.
    
    Params:
      entity_list (list[str]): A list of entities that appears in the environment.
    
    Returns:
      dict[str, list[float]]: A dictionary object matching each entity to
      an appearance.
    """
    entity_map: dict[str, list[float]] = {}
    num_classes = len(entity_list)
    for i, x in enumerate(entity_list):
      if x == 'EmptyEntity':
        entity_map[x] = np.zeros(num_classes)
      else:
        entity_map[x] = one_hot_encode(value=i, num_classes=num_classes)
    return entity_map
  
  def observe(
      self, 
      env: GridworldEnv,
      location: tuple | None = None,
  ) -> np.ndarray:
    r"""
    Basic environment observation function.

    Params:
      env: (GridworldEnv): The environment to observe.
      location: (Optional, tuple) - The location of the observer.
      If blank, returns the full environment.

    Returns:
      np.ndarray: The one-hot coded observation 
    
    Notes:
      If the observer's :code:`vision_radius` parameter is also `None`,
      this function will also return the full environment.
    """
    return visual_field(
      env=env,
      entity_map=self.entity_map,
      vision=self.vision_radius,
      location=location
    )
  
  def override_entity_map(
      self,
      entity_map: dict[str, list[float]]
  ) -> None:
    """Override the automatically generated entity map with a provided custom one.
    Can be useful if multiple classes should have the same appearance.
    
    Params:
      entity_map (dict[str, list[float]]): The new entity map to use."""
    self.entity_map = entity_map
  


  