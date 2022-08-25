
from gem.utils import (
    findInstance,
    one_hot,
    updateEpsilon,
    updateMemories,
    transferMemories,
    findMoveables,
    findAgents,
    transferWorldMemories,
)


# replay memory class

from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.randomActions import modelRandomAction
from models.cnn_lstm_dqn import model_CNN_LSTM_DQN


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
from gemworld.gemsWolves import WolfsAndGems


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



def playGame(
    models,
    trainableModels,
    worldSize=15,
    epochs=200000,
    maxEpochs=100,
    epsilon=0.9,
    gameVersion = WolfsAndGems
):

    losses = 0
    gamePoints = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25


    for epoch in range(epochs):
        env = gameVersion()

        done = 0
        withinTurn = 0

        moveList = findMoveables(env.world)
        for i, j in moveList:
            env.world[i, j, 0].init_replay(5)

        while done == 0:

            findAgent = findAgents(env.world)
            if len(findAgent) == 0:
                done = 1

            withinTurn = withinTurn + 1
            turn = turn + 1

            # this may be a better form than having functions that do nothing in a class
            if turn % sync_freq == 0:
                for mods in trainableModels:
                    models[mods].model2.load_state_dict(
                        models[mods].model1.state_dict()
                    )
                    # models[mods].updateQ

            moveList = findMoveables(env.world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                env.world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = env.world[i, j, 0]

                # note the prep vision may need to be a function within the model class
                input = models[holdObject.policy].createInput(env.world, i, j, holdObject)

                if holdObject.static != 1:
                    # I assume that we will need to update the "action" below to be something like
                    # [output] where action is the first thing that is returned
                    # the current structure would not work with multi-head output (Actor-Critic, immagination, etc.)
                    action = models[holdObject.policy].takeAction([input, epsilon])

                if withinTurn == maxEpochs:
                    done = 1

            # rewrite this so all classes have transition, most are just pass
            if holdObject.has_transitions == True:
                env.world, models, gamePoints = holdObject.transition(
                    action,
                    env.world,
                    models,
                    i,
                    j,
                    gamePoints,
                    done,
                    input,
                )

            # transfer the events for each agent into the appropriate model after all have moved
            expList = findMoveables(env.world)
            env.world = updateMemories(models, env.world, expList, endUpdate=True)

            # expList = findMoveables(world)
            stableVersion = False
            if stableVersion == False:
                models = transferWorldMemories(models, env.world, expList)
            if stableVersion == True:
                models = transferMemories(models, env.world, expList)

            # testing training after every event
            if withinTurn % modelUpdate_freq == 0:
                for mods in trainableModels:
                    loss = models[mods].training(150, 0.9)
                    losses = losses + loss.detach().numpy()

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = updateEpsilon(epsilon, turn, epoch)

        if epoch % 100 == 0:
            print(epoch, withinTurn, gamePoints, losses, epsilon)
            gamePoints = [0, 0]
            losses = 0
    return models


def train_wolf_gem(epochs=10000):
    models = []
    # 405 / 1445 should go back to 650 / 2570 when fixed
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 650, 350, 100, 4))  # agent model
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        15,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        0.85,  # starting epsilon
        gameVersion = WolfsAndGems,
    )
    return models

models = train_wolf_gem(1000)
