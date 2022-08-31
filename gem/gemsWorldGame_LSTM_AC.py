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
from models.cnn_lstm_AC import model_CNN_LSTM_AC


from models.perception import agentVisualField

"""
TODO: Remove old/stale imports.
"""

# from environment.elements import (
#     Agent,
#     EmptyObject,
#     Wolf,
#     Gem,
#     Wall,
#     deadAgent,
# )
from old_game_file.game_utils import createWorld, createWorldImage

# from gemworld.gemsWolves import WolfsAndGems
from gemworld.gemsWolvesSingle import WolfsAndGemsSingle


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
    gameVersion=WolfsAndGemsSingle(),
    trainModels=True,
):

    losses = 0
    gamePoints = [0, 0]
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25
    env = WolfsAndGemsSingle()

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
            env.world[i, j, 0].AC_logprob = []
            env.world[i, j, 0].AC_value = []
            env.world[i, j, 0].AC_reward = []

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

                    # note the prep vision may need to be a function within the model class
                    input = models[holdObject.policy].createInput(
                        env.world, i, j, holdObject
                    )

                    # I assume that we will need to update the "action" below to be something like
                    # [output] where action is the first thing that is returned
                    # the current structure would not work with multi-head output (Actor-Critic, immagination, etc.)
                    output = models[holdObject.policy].takeAction(input)
                    action, logprob, value = output

                    env.world[i, j, 0].AC_logprob.append(logprob)
                    env.world[i, j, 0].AC_value.append(value)
                    # env.world[i, j, 0].AC_reward = []

                    # problem - we want to get this into the replay buffer, but the replay buffer is hard coded into the
                    # replay for DQN. Maybe we could add a new memory class to the object at the beginning of this program?

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

                for i, j in expList:
                    env.world[i, j, 0].AC_reward.append(env.world[i, j, 0].reward)

                # expList = findMoveables(world)
                # modelType = "AC"
                # if modelType == "DQN":
                #    models = transferWorldMemories(models, env.world, expList)
                # if modelType == "AC":
                #    models[holdObject.policy].transferMemories_AC(holdObject.reward)

                # testing training after every event
                # if withinTurn % modelUpdate_freq == 0:
                #    for mods in trainableModels:
                #        loss = models[mods].training(150, 0.9)
                #        losses = losses + loss.detach().numpy()

        # agentList = []
        # for i in range(env.world.shape[0]):
        #    for j in range(env.world.shape[0]):
        #        if (
        #            env.world[i, j, 0].kind == "agent"
        #            or env.world[i, j, 0].kind == "deadAgent"
        #        ):
        #            agentList.append([i, j])
        if trainModels == True:
            expList = findMoveables(env.world)
            for i, j in expList:

                # need to put into one big model file and then run once to avoid
                # going through graph more than once?
                # i, j = expList[0]
                if env.world[i, j, 0].policy < 2:
                    clc = 0.1
                    gamma = 0.95
                    # gamma = 0.8
                    rewards = (
                        torch.Tensor(env.world[i, j, 0].AC_reward)
                        .flip(dims=(0,))
                        .view(-1)
                    )  # A
                    logprobs = (
                        torch.stack(env.world[i, j, 0].AC_logprob)
                        .flip(dims=(0,))
                        .view(-1)
                    )
                    values = (
                        torch.stack(env.world[i, j, 0].AC_value)
                        .flip(dims=(0,))
                        .view(-1)
                    )
                    Returns = []
                    ret_ = torch.Tensor([0])
                    for r in range(rewards.shape[0]):  # B
                        ret_ = rewards[r] + gamma * ret_
                        Returns.append(ret_)
                    Returns = torch.stack(Returns).view(-1)
                    Returns = F.normalize(Returns, dim=0)
                    actor_loss = -1 * logprobs * (Returns - values.detach())  # C
                    critic_loss = torch.pow(values - Returns, 2)  # D
                    loss = actor_loss.sum() + clc * critic_loss.sum()  # E
                    # models[env.world[i, j, 0].policy].optimizer.zero_grad() # this wasn't in example
                    loss.backward()
                    # self.optimizer.step()
                    models[env.world[i, j, 0].policy].optimizer.step()

        # then, we move the individual memories to the main memory of the model here

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
        25,  # world size
        1,  # number of epochs
        100,  # max epoch length
        0.85,  # starting epsilon
        gameVersion=WolfsAndGemsSingle,  # which game
        trainModels=False,  # this plays a game without learning
    )
    ani1.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 25, video_num, WolfsAndGemsSingle, vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def train_wolf_gem(epochs=10000, epsilon=0.85):
    models = []
    # 405 / 1445 should go back to 650 / 2570 when fixed
    # models.append(model_CNN_LSTM_AC(5, 0.00005, 1500, 650, 300, 75, 4))  # agent model
    models.append(model_CNN_LSTM_AC(5, 0.0001, 1500, 650, 100, 50, 4))  # agent model
    # models.append(model_CNN_LSTM_AC(5, 0.00001, 1500, 650, 300, 75, 4))  # agent model
    # models.append(model_CNN_LSTM_AC(5, 0.00001, 1500, 2570, 300, 75, 4))  # wolf model
    # models.append(model_CNN_LSTM_AC(5, 0.00001, 1500, 2570, 300, 75, 4))  # wolf model
    models = playGame(
        models,  # model file list
        [0],  # which models from that list should be trained, here not the agents
        25,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        0.85,  # starting epsilon
        gameVersion=WolfsAndGemsSingle,
    )
    return models


def addTrain_wolf_gem(models, epochs=10000, epsilon=0.3):
    models = playGame(
        models,  # model file list
        [0],  # which models from that list should be trained, here not the agents
        25,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        epsilon,  # starting epsilon
        gameVersion=WolfsAndGemsSingle,
    )
    return models


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
models = train_wolf_gem(5000)
save_models(models, save_dir, "acmodelClass_test_5000_do", 5)

models = addTrain_wolf_gem(models, 5000, 0.7)
save_models(models, save_dir, "acmodelClass_test_10000_do", 5)
