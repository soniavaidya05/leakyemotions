# ----------------------------------------------------- #
#        Abstract class for environment objects         #
# ----------------------------------------------------- #

class Object:
    '''
    Base element class. Defines the non-optional initialization parameters for all entities.
    '''
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

    Cuisine: specifies the name of the truck.
    '''
    def __init__(self, color, value, cuisine = 'generic'):
        super().__init(color)
        self.value = value # Value is specified in advance
        self.passable = True # You eat the food by stepping on top of the truck.
        self.cuisine = cuisine

    def __str__(self):
        return self.cuisine + str(self.__class__.__name__)

# endregion
# ----------------------------------------------------- #