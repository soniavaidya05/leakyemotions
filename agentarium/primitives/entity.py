from __future__ import annotations

from typing import Any, Optional

from agentarium.location import Location

# ----------------------------------------------------- #
#        Abstract class for environment objects         #
# ----------------------------------------------------- #


class Entity:
    r"""
    Base element class. Defines the non-optional initialization parameters for all entities.

    Attributes:
        - :attr:`appearance` - The appearance of the object.
        - :attr:`location` - The location of the object. It may take on the value of None when the Entity is first initialized.
        - :attr:`value` - The reward provided to an agent upon interaction. It is 0 by default.
        - :attr:`passable` - Whether the object can be traversed by an agent. It is False by default.
        - :attr:`has_transitions` - Whether the object has unique physics interacting with the environment. It is False by default.
        - :attr:`kind` - The class string of the object.
    """

    appearance: Any
    location: Optional[Location]
    value: float
    passable: bool
    has_transitions: bool
    kind: str

    def __init__(self, appearance):
        self.appearance = appearance  # Every object needs an appearance
        self.location = None
        self.value = 0  # By default, entities provide no reward to agents
        self.passable = (
            False  # Whether the object can be traversed by an agent (default: False)
        )
        self.has_transitions = False  # Entity's environment physics
        self.kind = str(self)

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return f"{self.__class__.__name__}(appearance={self.appearance},value={self.value})"

    def transition(self, env):
        """
        Change the environment in some way.
        By default, this function does nothing.

        Args:
            env (GridWorldEnv): the environment to enact transition to.
        """
        pass  # Entities do not have a transition function by default
