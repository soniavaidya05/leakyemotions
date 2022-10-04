from gem.environment.elements.element import Element


class Gem(Element):

    kind = "gem"  # class variable shared by all instances

    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearence = color  # gems are green
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.reward = 0  # how much reward this gem has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"
