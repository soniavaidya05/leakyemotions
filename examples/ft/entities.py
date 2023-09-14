# ----------------------------------------------------- #
#        Abstract class for environment objects         #
# ----------------------------------------------------- #

class Object:
    """
    Base element class. Defines the non-optional initialization parameters for all entities.

    Parameters:
        color: The color of the object.

    Attributes:
        appearance: The appearance of the object, specified by color.
        vision: The ability of the object to see an N x N pixels around it.
        value: The reward provided to an agent upon interaction.
        model: The neural network of the object.
        passable: Whether the object can be traversed by an agent.
        has_transitions: Whether the object has unique physics interacting with the environment.
        kind: The class string of the object.
    """
    def __init__(self, color):
        self.appearance = color # Every object needs an appearance
        self.vision = 0 # By default, entities cannot see
        self.value = 0 # By default, entities provide no reward to agents
        self.model = None # By default, entities have no transition policy
        self.passable = False # Whether the object can be traversed by an agent (default: False)
        self.has_transitions = False # Entity's environment physics
        self.kind = str(self)

    def __str__(self):
        return str(self.__class__.__name__)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(color={self.appearance},value={self.value})'

# ----------------------------------------------------- #
# region: Environment object classes for Baker ToM task #
# ----------------------------------------------------- #

class EmptyObject(Object):
    '''
    Base empty object.
    '''
    def __init__(self, color):
        super().__init__(color)
        self.passable = True # EmptyObjects can be traversed

class Wall(Object):
    '''
    Base wall object.
    '''
    def __init__(self, color):
        super().__init__(color)
        self.value = -1 # Walls penalize contact

class Truck(Object):
    '''
    Base truck object.

    Parameters:
        color: The appearance of the truck. \n
        cfg: The configuration object.

    Attributes:
        cuisine: specifies the name of the truck.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.cfg = cfg
        self.value = cfg.value # Value is specified in advance
        self.passable = True # You eat the food by stepping on top of the truck.
        self.kind = cfg.cuisine
        self.cuisine = cfg.cuisine
        self.done = True

    def __repr__(self):
        return f'{self.__class__.__name__}(color={self.appearance},value={self.value},cuisine={self.cuisine})'

# ----------------------------------------------------- #
# endregion                                             #
# ----------------------------------------------------- #