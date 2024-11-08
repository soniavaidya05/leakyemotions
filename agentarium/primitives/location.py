
from __future__ import annotations
from vector import Vector

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
