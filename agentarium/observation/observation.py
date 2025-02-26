import abc
import numpy as np

from agentarium.observation.visual_field import visual_field
from agentarium.utils.helpers import one_hot_encode
from agentarium.environments import GridworldEnv

class ObservationSpec:
    r"""
    An abstract class of an object that contains the observation specifications for Agentarium agents.

    Attributes:
        - :attr:`entity_map` - A mapping of the kinds of entities in the environment to their appearances.
        - :attr:`vision_radius` - The radius of the agent's vision. If None, the agent can see the entire environment.
    """
    entity_map: dict[str, ]
    vision_radius: int | None

    def __init__(
        self,
        entity_list: list[str],
        vision_radius: int | None = None
    ):
        r"""
        Initialize the ObservationSpec object.
        This function uses generate_map() to create an entity map for the ObservationSpec based on entity_list. 

        Args:
            entity_list (list[str]): A list of the kinds of entities that appear in the environment.
            vision_radius (Optional, int): The radius of the agent's vision. Defaults to None.
        """
        if not isinstance(vision_radius, int):
            self.vision_radius = None
        else:
            self.vision_radius = vision_radius
        self.entity_map = self.generate_map(entity_list)

    @abc.abstractmethod
    def generate_map(self, entity_list: list[str]) -> dict[str, ]:
        r"""
        Given a list of entities, return a dictionary of appearance values
        for the agent's observation function.
        
        Args:
            entity_list (list[str]): A list of the kinds of entities that appears in the environment.
        
        Returns:
            dict[str, list[float]]: A dictionary object matching each entity kind to
            an appearance.
        """
        pass
  
    @abc.abstractmethod
    def observe(
        self, 
        env: GridworldEnv,
        location: tuple | None = None,
    ) -> np.ndarray:
        r"""
        Basic environment observation function.

        Args:
            env: (GridworldEnv): The environment to observe.
            location: (Optional, tuple) - The location of the observer.
            If None, returns the full environment.

        Returns:
            np.ndarray: The observation.
        
        Notes:
            If :attr:`vision_radius` is also `None`,
            this function will also return the full environment.
        """
        pass
    
    def override_entity_map(
        self,
        entity_map: dict[str, ]
    ) -> None:
        r"""
        Override the automatically generated entity map from generate_map() with a provided custom one.
        Can be useful if multiple classes should have the same appearance.
        
        Args:
            entity_map (dict[str, ]): The new entity map to use.
        """
        self.entity_map = entity_map
    


class OneHotObservationSpec(ObservationSpec):
    """
    A subclass of :py:class:`ObservationSpec` for Agentarium agents whose observations take the form of one-hot encodings.
    
    Attributes:
        - :attr:`entity_map` - A mapping of the kinds of entities in the environment to their appearances.
        - :attr:`vision_radius` - The radius of the agent's vision. If None, the agent can see the entire environment.
    """
    entity_map: dict[str, list[float]]
    vision_radius: int | None

    def __init__(
        self,
        entity_list: list[str],
        vision_radius: int | None = None
    ):
        super.__init__(entity_list, vision_radius)

    def generate_map(self, entity_list: list[str]) -> dict[str, list[float]]:
        r"""
        Generate a default entity map by automatically creating one-hot encodings for each entity in the environment, 
        except for the "EmptyEntity" kind, which will receive an all-zero appearance.
        
        Args:
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
        Observes the environment using :py:func:`visual_field`.

        Args:
            env: (GridworldEnv): The environment to observe.
            location: (Optional, tuple) - The location of the observer.
            If blank, returns the full environment.

        Returns:
            np.ndarray: The one-hot coded observation, in the shape of `(number of channels, 2 * vision + 1, 2 * vision + 1)`.
            If :attr:`vision_radius` is `None` or the :code:`location` parameter is None, 
            the shape will be `(number of channels, env.width, env.layers)`.
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
    