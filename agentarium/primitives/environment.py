from __future__ import annotations

import numpy as np

from agentarium.location import Location
from agentarium.primitives.entity import Entity


class GridworldEnv:
    r"""
    Basic gridworld environment class with functions for getting and manipulating the locations of entities.

    Attributes:
        - :attr:`height` - The height of the gridworld.
        - :attr:`width` - The width of the gridworld.
        - :attr:`layers` - The number of layers that the gridworld has.
        - :attr:`default_entity` - The entity that the gridworld is filled with at creation by default.
        - :attr:`world` - A representation of the gridworld as a Numpy array of Entities, with dimensions height x width x layers.
    """

    height: int
    width: int
    layers: int
    default_entity: Entity

    world: np.array

    def __init__(self, height: int, width: int, layers: int, default_entity: Entity):
        self.height = height
        self.width = width
        self.layers = layers
        self.default_entity = default_entity
        self.create_world()

    def create_world(self):
        """
        Create a gridworld of dimensions H x W x L.
        """

        self.world = np.full(
            (self.height, self.width, self.layers), Entity(appearance=[0.0, 0.0, 0.0])
        )

        # Define the location of each entity
        for index, x in np.ndenumerate(self.world):
            self.world[index] = self.default_entity
            self.world[index].location = index

    # --------------------------- #
    # region: helper functions    #
    # --------------------------- #

    def add(self, target_location, entity: Entity) -> None:
        """
        Adds an entity to the world at a location.
        Will replace any existing entity at that location.

        Args:
            target_location (Location): the location of the entity
            entity (Entity): the entity to be added
        """
        entity.location = target_location
        self.world[target_location] = entity

    def remove(self, target_location) -> Entity:
        """
        Remove the entity at a location.

        Args:
            target_location (Location): the location of the entity

        Returns:
            Entity: the entity previously at the given location
        """
        entity = self.world[target_location]
        self.world[target_location] = self.default_entity
        self.world[target_location].location = target_location
        return entity

    def move(self, entity, new_location) -> bool:
        """
        Move an entity to a new location.

        Args:
            entity: entity to be moved
            new_location: location to move the entity to

        Returns:
            bool: True if move was successful (i.e. the entity currently at new_location is passable), False otherwise
        """
        if self.world[new_location].passable:
            self.remove(new_location)
            previous_location = entity.location
            entity.location = new_location
            self.world[new_location] = entity
            self.world[previous_location] = self.default_entity
            self.world[previous_location].location = previous_location
            return True
        else:
            return False

    def observe(self, target_location) -> Entity:
        """
        Observes the entity at a location.
        """
        return self.world[target_location]

    def valid_location(self, index: tuple[int, ...] | Location) -> bool:
        """
        Check if the given index is in the world.

        Args:
            index(tuple[int, ...] | Location): A tuple of coordinates or a Location object.

        Returns:
            bool: Whether the index is in env.world."""
        # Cast to tuple if it is a location
        if isinstance(index, Location):
            index = index.to_tuple()
        # Get world shape
        shape = self.world.shape
        # Can't compare if the tuples are of unequal size
        if len(index) != len(shape):
            raise IndexError(
                f"Index {index} and world shape {shape} must be the same length."
            )
        # Indices of less than 0 are not valid locations on a grid
        if min(index) < 0:
            return False
        # Otherwise, check if the index value exceeds the world shape on any dimension.
        for x, y in zip(shape, index):
            if y >= x:
                return False
        return True

    # TODO: formalize this function
    def get_entities(self, entity, locs=False) -> list[Entity]:
        """
        Return a list of entities or a list of their locations in the world.
        """
        entities = []
        for index, x in np.ndenumerate(self.world):
            if x.kind == entity:
                if locs:
                    entities.append(index)
                else:
                    entities.append(x)
        return entities

    @staticmethod
    def get_entities_(world, entity, locs=False) -> list[Entity]:
        """
        Return a list of entities or a list of their locations in the world.
        """
        entities = []
        for index, x in np.ndenumerate(world):
            if x.kind == entity:
                if locs:
                    entities.append(index)
                else:
                    entities.append(x)
        return entities

    # --------------------------- #
    # endregion: helper functions #
    # --------------------------- #
