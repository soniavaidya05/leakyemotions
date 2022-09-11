from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModelRandomAction:

    kind = "randomAction"  # class variable shared by all instances

    def __init__(self, replaySize, outsize):

        self.replay = deque([], maxlen=replaySize)
        self.sm = nn.Softmax(dim=1)
        self.actionSpace = outsize

    def takeAction(self, params):
        inp, epsilon = params
        action = np.random.randint(0, self.actionSpace)
        return action

    def training(self, batch_size, gamma):
        # this model does not train
        loss = torch.tensor(0.0)
        return loss

    def updateQ(self):
        pass
