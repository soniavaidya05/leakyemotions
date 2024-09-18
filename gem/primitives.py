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
    
    def valid_location(self, index: tuple[int, ...] | Location) -> bool:
        """Check if the given index is in the world.
        
        Parameters:
            index: A tuple of coordinates or Location.
        
        Return:
            bool: Whether the index is in env.world."""
        # Cast to tuple if it is a location
        if isinstance(index, Location):
            index = index.to_tuple()
        # Get world shape
        shape = self.world.shape
        # Can't compare if the tuples are of unequal size
        if len(index) != len(shape):
            raise IndexError(f"Index {index} and world shape {shape} must be the same length.")
        # Indices of less than 0 are not valid locations on a grid
        if min(index) < 0:
            return False
        # Otherwise, check if the index value exceeds the world shape on any dimension.
        for x, y in zip(shape, index):
            if y >= x:
                return False
        return True
    
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
        
    def __eq__(self, other) -> bool:
        """Equality"""

        # Compare a location
        if isinstance(other, Location):
            return True if ((self.x == other.x) and (self.y == other.y) and (self.z == other.z)) else False
        
        # Compare a vector    
        elif isinstance(other, Vector):
            return self == other.compute()
        
        # Compare a tuple
        elif isinstance(other, tuple):
            if len(other) == 2:
                return True if ((self.x == other[0]) and (self.y == other[1]) and (self.dims == 2)) else False
            else:
                return True if ((self.x == other[0]) and (self.y == other[1]) and (self.z == other[2])) else False
            
        # Unimplemented
        else:
            raise NotImplementedError
            
class Vector:
    def __init__(self, forward: int, right: int, backward: int = 0, left: int = 0, layer = 0, direction: int = 0): # Default direction: 0 degrees / UP / North
        """
        Initialize a vector object.
        
        Parameters:
            forward: (int) The number of steps forward.
            right: (int) The number of steps right.
            backward: (int, Optional) The number of steps backward. Since negative vectors are supported, this can be carried by the forward value.
            left: (int, Optional) The number of steps left. Since negative vectors are supported, this can be carried by the right value.
            layer: (int, Optional) The number of layers up (positive) or down (negative).
            direction: (int, default = 0) A compass direction. 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST.
            """
        self.direction = direction
        self.forward = forward
        self.right = right
        self.backward = backward
        self.left = left
        self.layer = layer

    def __repr__(self):
        return f"Vector(direction={self.direction},forward={self.forward},right={self.right},backward={self.backward},left={self.left}"
    
    def __str__(self):
        return repr(self)
    
    def __mul__(self, other) -> Vector:
        """Multiply a location by an integer amount."""

        if isinstance(other, int):
            return Vector(self.forward * other, self.right * other, self.backward * other, self.left * other, self.layer * other, self.direction)
        
        # Unimplemented
        else:
            raise NotImplementedError
        
    def __add__(self, other) -> Vector:
        """Add two vectors together. The vectors must be with respect to the same direction."""
        if isinstance(other, Vector):
            # Rotate the vector to match the current direction.
            other.rotate(self.direction)
            return Vector(
                self.forward + other.forward, 
                self.right + other.right, 
                self.backward + other.backward, 
                self.left + other.left, 
                self.layer + other.layer, 
                direction=self.direction
            )
        else:
            raise NotImplementedError
        
    def rotate(self, new_direction: int) -> Vector:
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
        
        return (forward * self.forward) + (right * self.right) + (backward * self.backward) + (left * self.left) + Location(0, 0, self.layer)
    
    def to_tuple(self) -> tuple[int, ...]:
        return self.compute().to_tuple()

            

        