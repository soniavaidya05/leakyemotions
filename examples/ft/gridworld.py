from examples.ft.entities import Object
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
            fill_value = self.default_object
        )

    # --------------------------- #
    # region: helper functions    #
    # --------------------------- #

    def add(self, target_location, object):
        '''
        Adds an object to the world at a location
        Will replace any existing object at that location
        '''
        self.world[target_location] = object

    def remove(self, target_location) -> Object:
        '''
        Remove the object at a location and return it

        Args:
            target_location: the location of the object
        '''
        object = self.world[target_location]
        self.world[target_location] = self.default_object
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