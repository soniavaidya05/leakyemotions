from collections import deque
import numpy as np
import torch



class Agent():
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"

    def generate_alien(self):
        alien_type = np.random.choice([0,1,2,3,4,5])
        if alien_type == 0:
            appearance = [0, np.random.choice([0,1]), 0, np.random.choice([0,1]), 0, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]) ]
            cooperation = np.random.choice([-1,1], p = (.1, .9))
        if alien_type == 1:
            appearance = [0, np.random.choice([0,1]), 1, np.random.choice([0,1]), 1 , np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            cooperation = np.random.choice([-1,1], p = (.3, .7))
        if alien_type == 2:
            appearance = [0, np.random.choice([0,1]), 0, np.random.choice([0,1]), 1, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]) ]
            cooperation = np.random.choice([-1,1], p = (.6, .4))
        if alien_type == 3:
            appearance = [1, 1, np.random.choice([0,1]), np.random.choice([0,1]), 1 , np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            cooperation = np.random.choice([-1,1], p = (.9, .1))
        if alien_type == 4:
            appearance = [1, 0, np.random.choice([0,1]), np.random.choice([0,1]), 1, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]) ]
            cooperation = np.random.choice([-1,1], p = (.7, .3))
        if alien_type == 5:
            appearance = [1, 0, np.random.choice([0,1]), np.random.choice([0,1]), 0 , np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1]),np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            cooperation = np.random.choice([-1,1], p = (.4, .6))
        return alien_type, appearance, cooperation


    def transition(self, action, cooperation, condition = "full"):
        if condition == "full":
            if action == 0:
                reward = cooperation * -1
            if action == 1:
                reward = cooperation
        if condition == "partial":
            if action == 0:
                reward = 0
            if action == 1:
                reward = cooperation

        return reward

