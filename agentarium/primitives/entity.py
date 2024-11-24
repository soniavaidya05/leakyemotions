from __future__ import annotations

# ----------------------------------------------------- #
#        Abstract class for environment objects         #
# ----------------------------------------------------- #


class Entity:
    r"""
    Base element class. Defines the non-optional initialization parameters for all entities.

    Parameters:
        appearance: The color of the object.

    Attributes:
        appearance: The appearance of the object.
        value: The reward provided to an agent upon interaction.
        passable: Whether the object can be traversed by an agent.
        has_transitions: Whether the object has unique physics interacting with the environment.
        kind: The class string of the object.
    """

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
        """ """
        pass  # Entities do not have a transition function by default
