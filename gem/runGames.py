#!/usr/bin/env python
# coding: utf-8

# from gem.utils import (
#    findInstance,
#    one_hot,
#    updateEpsilon,
#    updateMemories,
#    transferMemories,
#    findMoveables,
#    findAgents,
# )


# replay memory class

# from models.memory import Memory
# from models.dqn import DQN, modelDQN
# from models.randomActions import modelRandomAction

# from models.perception import agentVisualField
# from environment.elements import (
#    Agent,
#    EmptyObject,
#    Wolf,
#    Gem,
#    Wall,
#    deadAgent,
# )
# from gem.game_utils import createWorld, createWorldImage
# from gemworld.transitions import agentTransitions, wolfTransitions


# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from astropy.visualization import make_lupton_rgb

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import random
# import copy

# import pickle

# from collections import deque

# from gemGame import runCombinedTraining, moreTraining
from gemGame_experimental import train_wolf_gem, save_models, load_models

# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

models = train_wolf_gem(50000)
save_models(models, save_dir, "wolf_gem_50000", 10)


# it may be necessary to initialize a large game with all trainable objects in it once
# for 5 turns to make sure everything is setup. Ideally we would not need that
# but as a temp looks better than failing

# need to figure out why wolfeat is not being updated
# i figured it out..... wolf attack is not being passed back
# in the transitions arguments!
