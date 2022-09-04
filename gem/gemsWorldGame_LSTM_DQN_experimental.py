from utils import (
    findInstance,
    one_hot,
    updateEpsilon,
    updateMemories,
    transferMemories,
    findMoveables,
    findAgents,
    transferWorldMemories,
)

from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.randomActions import modelRandomAction
from models.cnn_lstm_dqn import model_CNN_LSTM_DQN

from models.perception import agentVisualField

from game_utils import createWorld, createWorldImage
from gemworld.gemsWolvesLargeWorld import WolfsAndGems

import os

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
    gameVersion=WolfsAndGems(),
    trainModels=True,
):

    losses = 0
    gamePoints = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 10
    env = WolfsAndGems()

    if trainModels == False:
        fig = plt.figure()
        ims = []

    for epoch in range(epochs):
        env.reset_env()

        done = 0
        withinTurn = 0

        moveList = findMoveables(env.world)
        for i, j in moveList:
            env.world[i, j, 0].init_replay(5)

        while done == 0:

            if trainModels == False:
                image = createWorldImage(env.world)
                im = plt.imshow(image, animated=True)
                ims.append([im])

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

                if holdObject.static != 1:

                    """
                    Currently RNN and non-RNN models have different createInput files, with
                    the RNN having createInput and createInput2. This needs to be fixed

                    This creates an agent specific view of their environment
                    This also may become more challenging with more output heads

                    """

                    inputs = models[holdObject.policy].createInput2(
                        env.world, i, j, holdObject, 1
                    )
                    input, combined_input = inputs

                    """
                    Below generates an action

                    """

                    action = models[holdObject.policy].takeAction(
                        [combined_input, epsilon]
                    )

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

            if trainModels == True:
                # transfer the events for each agent into the appropriate model after all have moved
                expList = findMoveables(env.world)
                env.world = updateMemories(models, env.world, expList, endUpdate=True)

                # expList = findMoveables(world)
                modelType = "DQN"
                if modelType == "DQN":
                    models = transferWorldMemories(models, env.world, expList)
                if modelType == "AC":
                    models[holdObject.policy].transferMemories_AC(holdObject.reward)

                # testing training after every event
                if withinTurn % modelUpdate_freq == 0:
                    for mods in trainableModels:
                        loss = models[mods].training(300, 0.9)
                        losses = losses + loss.detach().numpy()

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = updateEpsilon(epsilon, turn, epoch)

        if epoch % 100 == 0 and trainModels == True:
            print(epoch, withinTurn, gamePoints, losses, epsilon)
            gamePoints = [0, 0]
            losses = 0
    if trainModels == True:
        return models
    if trainModels == False:
        ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True, repeat_delay=1000
        )
        return ani


def createVideo(models, worldSize, num, gameVersion, filename="unnamed_video.gif"):
    # env = gameVersion()
    ani1 = playGame(
        models,  # model file list
        [],  # which models from that list should be trained, here not the agents
        worldSize,  # world size
        1,  # number of epochs
        150,  # max epoch length
        0.1,  # starting epsilon
        gameVersion=WolfsAndGems,  # which game
        trainModels=False,  # this plays a game without learning
    )
    ani1.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 35, video_num, WolfsAndGems, vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def train_wolf_gem(epochs=10000, epsilon=0.85):
    models = []
    # 405 / 1445 should go back to 650 / 2570 when fixed
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 3000, 650, 425, 125, 4))  # agent model
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 3000, 2570, 425, 125, 4))  # wolf model
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        50,  # world size
        epochs,  # number of epochs
        400,  # max epoch length
        0.85,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


def addTrain_wolf_gem(models, epochs=10000, epsilon=0.3):
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        50,  # world size
        epochs,  # number of epochs
        400,  # max epoch length
        epsilon,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
models = train_wolf_gem(10000)
save_models(models, save_dir, "DQN_LSTM_10000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "DQN_LSTM_20000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.6)
save_models(models, save_dir, "DQN_LSTM_30000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.3)
save_models(models, save_dir, "DQN_LSTM_40000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.3)
save_models(models, save_dir, "DQN_LSTM_50000lw", 5)

# models = load_models(save_dir, "modelClass_test_20000")
