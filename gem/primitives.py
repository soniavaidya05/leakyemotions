from __future__ import annotations
import numpy as np

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

class Location:
    def __init__(self, *dims):
        """Initialize a Location object.
        
        Parameters:
            *dims: An unpacked tuple of coordinates. Supports up to three (x, y, z)."""
        self.dims = len(dims)
        self.x = dims[0]
        self.y = dims[1]
        if self.dims > 2:
            self.z = dims[2]
        else:
            self.z = 0

    def to_tuple(self) -> tuple[int, ...]:
        """Cast the Location back to a tuple."""
        if self.dims == 2:
            return (self.x, self.y)
        else:
            return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Location({self.x}, {self.y}, {self.z})"

    def __str__(self):
        return repr(self)

    def __add__(self, other) -> Location:
        """Add a location or vector.
        
        Params:
            other: An object of type Location or Vector.
            
        Return:
            Location: The new location."""
        
        # Add location
        if isinstance(other, Location):
            return Location(self.x + other.x, self.y + other.y, self.z + other.z)

        # Add a vector    
        elif isinstance(other, Vector):
            return self + other.compute()

        # Add a tuple
        elif isinstance(other, tuple):
            if self.dims == 2:
                return Location(self.x + other[0], self.y + other[1])
            elif len(other) == 2:
                return Location(self.x + other[0], self.y + other[1], self.z)
            else:
                return Location(self.x + other[0], self.y + other[1], self.z + other[2])
    
        # Unimplemented
        else:
            raise NotImplementedError
        
    def __mul__(self, other) -> Location:
        """Multiply a location by an integer amount."""

        if isinstance(other, int):
            return Location(self.x * other, self.y * other, self.z * other)
        
        # Unimplemented
        else:
            raise NotImplementedError

            
class Vector:
    def __init__(self, *urdl: list[int], direction: int = 0): # Default direction: 0 degrees / UP / North
        """
        Initialize a vector object.
        
        Parameters:
            *urdl: An unpacked tuple of ints indicating the number of steps forward, right, or (optionally) backward or left.
            Technically, only the first two are necessary, since negative vectors are supported.
            direction: (int) A compass direction. 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST.
            """
        self.direction = direction
        self.forward = urdl[0]
        self.right = urdl[1]
        if len(urdl) > 2:
            self.backward = urdl[2]
            self.left = urdl[3]
        else:
            self.backward = 0
            self.left = 0

    def __repr__(self):
        return f"Vector(direction={self.direction},forward={self.forward},right={self.right},backward={self.backward},left={self.left}"
    
    def __str__(self):
        return repr(self)
    
    def __mul__(self, other) -> Vector:
        """Multiply a location by an integer amount."""

        if isinstance(other, int):
            return Vector(self.forward * other, self.right * other, self.backward * other, self.left * other, self.direction)
        
        # Unimplemented
        else:
            raise NotImplementedError
        
    def __add__(self, other) -> Vector:
        """Add two vectors together. The vectors must be with respect to the same direction."""
        if isinstance(other, Vector):
            assert self.direction == other.direction, "Vectors must have the same direction."
            return Vector(self.forward + other.forward, self.right + other.right, self.backward + other.backward, self.left + other.left, direction=self.direction)
        
    def rotate(self, new_direction: int) -> None:
        """Rotate the vector to face a new direction. """
        num_rotations = (self.direction - new_direction) % 4
        match(num_rotations):
            case 0:
                pass
            case 1:
                self.right, self.backward, self.left, self.forward = self.forward, self.right, self.backward, self.left
            case 2:
                self.backward, self.left, self.forward, self.right = self.forward, self.right, self.backward, self.left
            case 3:
                self.left, self.forward, self.right, self.backward = self.forward, self.right, self.backward, self.left
        self.direction = new_direction
     
    def compute(self) -> Location:
        """Given a direction being faced and a number of paces
        forward / right / backward / left, compute the location."""

        match(self.direction):
            case 0:  # UP
                forward, right, backward, left = (Location(-1, 0), Location(0, 1), Location(1, 0), Location(0, -1))
            case 1:  # RIGHT
                forward, right, backward, left = (Location(0, 1), Location(1, 0), Location(0, -1), Location(-1, 0))
            case 2:  # DOWN
                forward, right, backward, left = (Location(1, 0), Location(0, -1), Location(-1, 0), Location(0, 1))
            case 3:  # LEFT
                forward, right, backward, left = (Location(0, -1), Location(-1, 0), Location(0, 1), Location(1, 0))
        
        return (forward * self.forward) + (right * self.right) + (backward * self.backward) + (left * self.left)
    
    def to_tuple(self) -> tuple[int, ...]:
        return self.compute().to_tuple()

            

        