from abc import ABC


class Element(ABC):
    def __init__(self):

        self.appearence = None  # how to display this agent
        self.vision = None  # visual radius
        self.policy = None  # policy function or model
        self.value = None  # reward given to another agent
        self.reward = None  # reward received on this trial
        self.static = True  # whether the object gets to take actions or not
        self.passable = False  # whether the object blocks movement
        self.trainable = False  # whether there is a network to be optimized
        self.has_transitions = False


class EmptyObject:

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # empty stuff is basically empty
        self.appearence = [0.0, 0.0, 0.0]  # empty is well, blank
        self.vision = 1  # empty stuff is basically empty
        self.policy = "NA"  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.reward = 0  # empty stuff is basically empty
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # wall stuff is basically empty
        self.appearence = [153.0, 51.0, 102.0]  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
