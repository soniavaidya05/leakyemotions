from collections import deque
from random import random
import numpy as np

# this may no longer be needed


class Memory:
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode):
        self.memory.append(epsiode)

    # get multiple sequences of expereicnes from multiple episodes (stories) (each sequence from a distinct episode)
    def get_batch(self, bsize, time_step):
        sampled_episodes = random.sample(self.memory, bsize)
        batches = []
        for episode in sampled_episodes:
            while len(episode) + 1 - time_step < 1:
                episode = random.sample(self.memory, 1)[0]
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batches.append(episode[point : point + time_step])

        return batches
