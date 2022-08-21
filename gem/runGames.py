#!/usr/bin/env python
# coding: utf-8

from gem.utils import (
    findInstance,
    one_hot,
    updateEpsilon,
    updateMemories,
    transferMemories,
    findMoveables,
    findAgents,
)


# replay memory class

from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.randomActions import modelRandomAction

from models.perception import agentVisualField
from environment.elements import (
    Agent,
    EmptyObject,
    Wolf,
    Gem,
    Wall,
    deadAgent,
)
from gem.game_utils import createWorld, createWorldImage
from gemworld.transitions import agentTransitions, wolfTransitions


import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.visualization import make_lupton_rgb

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import copy

import pickle

from collections import deque

from gemGame import runCombinedTraining, moreTraining


# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput/"


models = runCombinedTraining(
    save_dir,
    wolf_model="Model_StableWolfAttack",
    agent_model="modelGemsSearch_0",
    trainableModels=[0],
    epochs=30000,
    max_epochs=100,
    epsilon=0.9,
    videoNum=1000,
)


with open(save_dir + "combinedModel_1002", "rb") as fp:
    models = pickle.load(fp)
filename = "test1.gif"
createVideo(models, 25, 1, gameVersion="wolvesGems", filename=filename)
filename = "test2.gif"
createVideo(models, 25, 2, gameVersion="wolvesGems", filename=filename)
filename = "test3.gif"
createVideo(models, 35, 2, gameVersion="wolvesGems", filename=filename)


for game in range(10):
    models = moreTraining(save_dir, models, [0], 25, 10000, 100, 0.3, game + 2000)


# note, to really test the model, we need to build test cases. like simple
# 9 x 9 worlds with a green and yellow gem
# or a wolf in a location, and look at the Q values to see if avoidance is being learned

# additional note, may need to have larger worlds to escape from wolves in
