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
        alien_type = np.random.choice([0,1])
        if alien_type == 0:
            appearence = [alien_type, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            cooperation = np.random.choice([-1,1], p = (.1, .9))
        if alien_type == 1:
            appearence = [alien_type, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            cooperation = np.random.choice([-1,1], p = (.9, .1))
        return alien_type, appearence, cooperation


    def transition(self, action, cooperation):

        if action == 0:
            reward = cooperation * -1
        if action == 1:
            reward = cooperation

        return reward

