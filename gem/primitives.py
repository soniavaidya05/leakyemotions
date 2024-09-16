from __future__ import annotations

# ----------------------------------------------------- #
#        Abstract class for environment objects         #
# ----------------------------------------------------- #

class Object:
    """
    Base element class. Defines the non-optional initialization parameters for all entities.

    Parameters:
        appearance: The color of the object.

    Attributes:
        appearance: The appearance of the object.
        vision: The ability of the object to see an N x N pixels around it.
        value: The reward provided to an agent upon interaction.
        model: The neural network of the object.
        passable: Whether the object can be traversed by an agent.
        has_transitions: Whether the object has unique physics interacting with the environment.
        kind: The class string of the object.
    """
    def __init__(self, appearance):
        self.appearance = appearance # Every object needs an appearance
        self.location = None
        self.vision = 0 # By default, entities cannot see
        self.value = 0 # By default, entities provide no reward to agents
        self.model = None # By default, entities have no transition policy
        self.passable = False # Whether the object can be traversed by an agent (default: False)
        self.has_transitions = False # Entity's environment physics
        self.kind = str(self)

    def __str__(self):
        return str(self.__class__.__name__)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(appearance={self.appearance},value={self.value})'
    
    def transition(self, env: GridworldEnv):
        pass # Entities do not have a transition function by default

import numpy as np

class GridworldEnv:
    '''
    Abstract gridworld environment class with basic functions
    '''
    def __init__(
        self,
        height: int,
        width: int,
        layers: int,
        default_object: Object
    ):
        self.height = height
        self.width = width
        self.layers = layers
        self.default_object = default_object
        self.create_world()

    def create_world(self):
        '''
        Create a gridworld of dimensions H x W x L.
        '''

        self.world = np.full(
            (self.height, self.width, self.layers),
            Object(appearance=[0., 0., 0.])
        )

        # Define the location of each object
        for index, x in np.ndenumerate(self.world):
            self.world[index] = self.default_object
            self.world[index].location = index

    # --------------------------- #
    # region: helper functions    #
    # --------------------------- #

    def add(self, target_location, object):
        '''
        Adds an object to the world at a location
        Will replace any existing object at that location
        '''
        object.location = target_location
        self.world[target_location] = object

    def remove(self, target_location) -> Object:
        '''
        Remove the object at a location and return it

        Args:
            target_location: the location of the object
        '''
        object = self.world[target_location]
        self.world[target_location] = self.default_object
        self.world[target_location].location = target_location
        return object

    def move(self, object, new_location) -> bool:
        '''
        Move an object from a location to a new location
        Return True if successful, False otherwise

        Args:
            object: object to be moved
            new_location: location to move the object
        '''
        if self.world[new_location].passable:
            self.remove(new_location)
            previous_location = object.location
            object.location = new_location
            self.world[new_location] = object
            self.world[previous_location] = self.default_object
            self.world[previous_location].location = previous_location
            return True
        else:
            return False
        
    def observe(self, target_location) -> Object:
        '''
        Observes the object at a location
        '''
        return self.world[target_location]
    
    def get_entities(self, entity, locs = False) -> list:
        '''
        Return a list of entities or a list of their locations in the world.
        '''
        entities = []
        for index, x in np.ndenumerate(self.world):
            if x.kind == entity:
                if locs:
                    entities.append(index)
                else:
                    entities.append(x)
        return entities
    
    @staticmethod
    def get_entities_(world, entity, locs = False) -> list:
        '''
        Return a list of entities or a list of their locations in the world.
        '''
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