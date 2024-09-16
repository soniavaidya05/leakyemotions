from gem.primitives import Object

# ----------------------------------------------------- #
# region: Environment object classes for Baker ToM task #
# ----------------------------------------------------- #

class EmptyObject(Object):
    '''
    Base empty object.
    '''
    def __init__(self, appearance):
        super().__init__(appearance)
        self.passable = True # EmptyObjects can be traversed

class Wall(Object):
    '''
    Base wall object.
    '''
    def __init__(self, appearance):
        super().__init__(appearance)
        self.value = -1 # Walls penalize contact

class Truck(Object):
    '''
    Base truck object.

    Parameters:
        appearance: The appearance of the truck. \n
        cfg: The configuration object.

    Attributes:
        cuisine: specifies the name of the truck.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.cfg = cfg
        self.value = cfg.value # Value is specified in advance
        self.passable = True # You eat the food by stepping on top of the truck.
        self.kind = cfg.cuisine
        self.cuisine = cfg.cuisine
        self.done = True

    def __repr__(self):
        return f'{self.__class__.__name__}(appearance={self.appearance},value={self.value},cuisine={self.cuisine})'

# ----------------------------------------------------- #
# endregion                                             #
# ----------------------------------------------------- #