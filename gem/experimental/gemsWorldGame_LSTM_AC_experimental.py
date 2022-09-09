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

from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.randomActions import modelRandomAction
from models.cnn_lstm_dqn import model_CNN_LSTM_DQN
from models.cnn_lstm_AC import model_CNN_LSTM_AC

from models.perception import agentVisualField

from game_utils import createWorld, createWorldImage

# from gemworld.gemsWolvesDual import WolfsAndGemsDual
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
    gameVersion=WolfsAndGems(),  # this is not working so hard coded below
    trainModels=True,
):

    losses = 0
    gamePoints = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25  # this is not needed for the current AC mdoel
    env = WolfsAndGems(worldSize, worldSize)

    if trainModels == False:
        fig = plt.figure()
        ims = []

    for epoch in range(epochs):
        env.reset_env(worldSize, worldSize)

        done = 0
        withinTurn = 0

        moveList = findMoveables(env.world)
        for i, j in moveList:
            # reset the memories for all agents
            env.world[i, j, 0].init_replay(5)
            env.world[i, j, 0].AC_logprob = torch.tensor([])
            env.world[i, j, 0].AC_value = torch.tensor([])
            env.world[i, j, 0].AC_reward = torch.tensor([])

        for mod in range(len(models)):
            """
            Resets the model memories to get ready for the new episode memories
            this likely should be in the model class when we figure out how to
            get AC and DQN models to have the same format
            """
            models[mod].rewards = torch.tensor([])
            models[mod].values = torch.tensor([])
            models[mod].logprobs = torch.tensor([])
            models[mod].Returns = torch.tensor([])

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

            # This is not needed for an Actor Critic model. Should cause a PASS
            if turn % sync_freq == 0:
                for mods in trainableModels:
                    models[mods].updateQ

            moveList = findMoveables(env.world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                env.world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = env.world[i, j, 0]

                if holdObject.static != 1:

                    inputs = models[holdObject.policy].createInput2(
                        env.world, i, j, holdObject, 2
                    )
                    input, combined_input = inputs

                    # I assume that we will need to update the "action" below to be something like
                    # [output] where action is the first thing that is returned
                    # the current structure would not work with multi-head output (Actor-Critic, immagination, etc.)
                    output = models[holdObject.policy].takeAction(combined_input)
                    action, logprob, value = output

                    # the lines below save the current memories of the event to
                    # the actor critic version of a replay. This should likely be
                    # in the model class rather than here

                    logprob = logprob.reshape(1, 1)

                    env.world[i, j, 0].AC_logprob = torch.concat(
                        [env.world[i, j, 0].AC_logprob, logprob]
                    )

                    env.world[i, j, 0].AC_value = torch.concat(
                        [env.world[i, j, 0].AC_value, value]
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
                        expBuff=True,
                        ModelType="AC",
                    )

            if trainModels == True:

                """
                transfer the events for each agent into the appropriate model after all have moved
                TODO: This needs to be rewritten as findTrainables, currently deadagents do not move
                    but they have information in their replay buffers that need to go into learning
                """

                expList = findMoveables(env.world)
                env.world = updateMemories(models, env.world, expList, endUpdate=True)
                for i, j in expList:
                    env.world[i, j, 0].AC_reward = torch.concat(
                        [
                            env.world[i, j, 0].AC_reward,
                            torch.tensor(env.world[i, j, 0].reward)
                            .float()
                            .reshape(1, 1),
                        ]
                    )

        if trainModels == True:

            expList = findMoveables(env.world)
            # TODO: just like above this needs to be changed to findTrainables because deadAgents have memories
            #       that need to be learned

            for i, j in expList:
                models[env.world[i, j, 0].policy].transferMemories_AC(env.world, i, j)

            for mod in range(len(models)):
                if len(models[mod].rewards) > 0:
                    loss = models[mod].training()
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
        0.85,  # starting epsilon
        gameVersion=WolfsAndGems,  # which game
        trainModels=False,  # this plays a game without learning
    )
    ani1.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 30, video_num, WolfsAndGems, vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def train_wolf_gem(epochs=10000, epsilon=0.85):
    models = []
    models.append(model_CNN_LSTM_AC(5, 0.00001, 1500, 650, 150, 75, 4))  # agent model
    models.append(model_CNN_LSTM_AC(5, 0.000001, 1500, 2570, 150, 75, 4))  # wolf model
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        30,  # world size
        epochs,  # number of epochs
        200,  # max epoch length
        0.85,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


def addTrain_wolf_gem(models, epochs=10000, epsilon=0.3):
    models = playGame(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        30,  # world size
        epochs,  # number of epochs
        200,  # max epoch length
        epsilon,  # starting epsilon
        gameVersion=WolfsAndGems,
    )
    return models


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
models = train_wolf_gem(1000)
save_models(models, save_dir, "AC_LSTM_1000lw", 5)

models = addTrain_wolf_gem(
    models, 9000, 0.7
)  # note, the epsilon pamamter is meaningless here
save_models(models, save_dir, "AC_LSTM_10000lw", 5)

models = addTrain_wolf_gem(
    models, 10000, 0.7
)  # note, the epsilon pamamter is meaningless here
save_models(models, save_dir, "AC_LSTM_20000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_30000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_40000lw", 5)

models = addTrain_wolf_gem(models, 10000, 0.7)
save_models(models, save_dir, "AC_LSTM_50000lw", 5)
