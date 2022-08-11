class Gem:

    kind = 'gem'                    # class variable shared by all instances

    def __init__(self, value, color):
        self.health = 1             # for the gen, whether it has been mined or not
        self.appearence = color    # gems are green
        self.vision = 1             # gems can see one radius around them
        self.policy = "NA"          # gems do not do anything
        self.value = value          # the value of this gem
        self.reward = 0             # how much reward this gem has found (will remain 0)
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 1           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized


class Agent:

    kind = 'agent'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [0.,0.,255.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = model         # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.epoch_mem = []
        self.CurrExp = CurrExp()
        self.memories = deque([],maxlen=3)



        
    def instanceDead(self):
        self.kind = "deadAgent"
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.appearence = [130.,130.,130.] # dead agents are grey
        self.epoch_mem = []
        # note, this has to allow for one last training
        

class deadAgent:

    kind = 'deadAgent'               # class variable shared by all instances

    def __init__(self):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [130.,130.,130.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = "NA"         # agent model here. 
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized
        # self.currentExp = []
        self.CurrExp = ()
        self.epoch_mem = []

class Wolf:

    kind = 'wolf'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [255.,0.,0.]    # agents are red
        self.vision = 4             # agents can see three radius around them
        self.policy = model         # gems do not do anything
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.epoch_mem = []
        self.currExp = CurrExp()
        self.memories = deque([],maxlen=3)


    def instanceDead(self):
        self.kind = "deadwolf"
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.appearence = [130.,130.,130.]    # dead agents are grey
        # note, this has to allow for one last training
        
class deadwolf:

    kind = 'deadwolf'               # class variable shared by all instances

    def __init__(self):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [130.,130.,130.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = "NA"         # agent model here. 
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized


class Wall:

    kind = 'wall'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0            # wall stuff is basically empty
        self.appearence = [153., 51., 102.]    # walls are purple
        self.vision = 0             # wall stuff is basically empty
        self.policy = "NA"          # walls do not do anything
        self.value = 0              # wall stuff is basically empty
        self.reward = -.1             # wall stuff is basically empty
        self.static = 1             # wall stuff is basically empty
        self.passable = 0           # you can't walk through a wall
        self.trainable = 0           # whether there is a network to be optimized

class BlastRay:

    kind = 'blastray'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0            
        self.appearence = [255., 255., 255.]    # blast rays are white
        self.vision = 0             # rays do not see
        self.policy = "NA"          # rays do not think
        self.value = 10              # amount of damage if you are hit by the ray
        self.reward = 0             # rays do not want
        self.static = 1             # rays exist for one turn
        self.passable = 1           # you can't walk through a ray without being blasted
        self.trainable = 0           # rays do not learn
        
        
class EmptyObject:

    kind = 'empty'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0             # empty stuff is basically empty
        self.appearence = [0.,0.,0.]  #empty is well, blank 
        self.vision = 1             # empty stuff is basically empty
        self.policy = "NA"          # empty stuff is basically empty
        self.value = 0              # empty stuff is basically empty
        self.reward = 0             # empty stuff is basically empty
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 1           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized

class tagAgent:

    kind = 'agent'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.is_it = 0              # everyone starts off not it
        self.appearence = [0., 0., 255.]    # agents are blue when not it
        self.vision = 4             # agents can see three radius around them
        # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.policy = model
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.frozen = 0

    def tag(self):
        if self.is_it == 0:
            self.is_it = 1
            self.appearence = [255, 0., 0.]
            self.frozen = 2
        else:
            self.is_it = 0
            self.appearence = [0., 0., 255]

