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
    transferMemories_LSTM,
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

# generate the gem search game objects

# generate the gem search game objects

agent1 = Agent(0)
wolf1 = Wolf(1)
gem1 = Gem(5, [0.0, 255.0, 0.0])
gem2 = Gem(15, [255.0, 255.0, 0.0])
gem3 = Gem(10, [0.0, 0.0, 255.0])
emptyObject = EmptyObject()
walls = Wall()

# create the instances
def createWolfHunt(worldSize, gem1p=0.115, gem2p=0.06, agent1p=0.05):

    # make the world and populate
    world = createWorld(worldSize, worldSize, 1, emptyObject)

    for i in range(worldSize):
        for j in range(worldSize):
            obj = np.random.choice(
                [0, 1, 2, 3], p=[gem1p, gem2p, agent1p, 1 - gem2p - gem1p - agent1p]
            )
            if obj == 0:
                world[i, j, 0] = gem1
            if obj == 1:
                world[i, j, 0] = gem2
            if obj == 2:
                world[i, j, 0] = agent1

    cBal = np.random.choice([0, 1])
    if cBal == 0:
        world[round(worldSize / 2), round(worldSize / 2), 0] = wolf1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = wolf1
    if cBal == 1:
        world[round(worldSize / 2), round(worldSize / 2), 0] = wolf1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = wolf1

    for i in range(worldSize):
        world[0, i, 0] = walls
        world[worldSize - 1, i, 0] = walls
        world[i, 0, 0] = walls
        world[i, worldSize - 1, 0] = walls

    return world


def createWolvesGems(worldSize, gem1p=0.115, gem2p=0.06, agent1p=0.005):

    # make the world and populate
    world = createWorld(worldSize, worldSize, 1, emptyObject)

    for i in range(worldSize):
        for j in range(worldSize):
            obj = np.random.choice(
                [0, 1, 2, 3], p=[gem1p, gem2p, agent1p, 1 - gem2p - gem1p - agent1p]
            )
            if obj == 0:
                world[i, j, 0] = gem1
            if obj == 1:
                world[i, j, 0] = gem2
            if obj == 2:
                world[i, j, 0] = wolf1

    cBal = np.random.choice([0, 1])
    if cBal == 0:
        world[round(worldSize / 2), round(worldSize / 2), 0] = agent1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = agent1
    if cBal == 1:
        world[round(worldSize / 2), round(worldSize / 2), 0] = agent1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = agent1

    for i in range(worldSize):
        world[0, i, 0] = walls
        world[worldSize - 1, i, 0] = walls
        world[i, 0, 0] = walls
        world[i, worldSize - 1, 0] = walls

    return world


def createGemsSearch(worldSize, gem1p=0.115, gem2p=0.06, agent1p=0.00):

    # make the world and populate
    world = createWorld(worldSize, worldSize, 1, emptyObject)

    for i in range(worldSize):
        for j in range(worldSize):
            obj = np.random.choice(
                [0, 1, 2, 3], p=[gem1p, gem2p, agent1p, 1 - gem2p - gem1p - agent1p]
            )
            if obj == 0:
                world[i, j, 0] = gem1
            if obj == 1:
                world[i, j, 0] = gem2
            if obj == 2:
                world[i, j, 0] = wolf1

    cBal = np.random.choice([0, 1])
    if cBal == 0:
        world[round(worldSize / 2), round(worldSize / 2), 0] = agent1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = agent1
    if cBal == 1:
        world[round(worldSize / 2), round(worldSize / 2), 0] = agent1
        world[round(worldSize / 2) + 1, round(worldSize / 2) - 1, 0] = agent1

    for i in range(worldSize):
        world[0, i, 0] = walls
        world[worldSize - 1, i, 0] = walls
        world[i, 0, 0] = walls
        world[i, worldSize - 1, 0] = walls

    return world


# test the world models


def gameTest(worldSize):
    world = createWolfHunt(worldSize)
    image = createWorldImage(world)

    moveList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].static == 0:
                moveList.append([i, j])

    img = agentVisualField(world, (moveList[0][0], moveList[0][1]), k=4)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()


# play and learn the game
def createInput(img):
    input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    return input


def playGame(
    models,
    trainableModels,
    worldSize=15,
    epochs=200000,
    maxEpochs=100,
    epsilon=0.9,
    gameVersion="wolfHunt",
):

    losses = 0
    gamePoints = [0, 0]
    status = 1
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25
    # note, rather than having random actions, we could keep the memories growing and just not train agents for
    # little while to train a different part of the model, or could stop training wolves, etc.

    for epoch in range(epochs):
        if gameVersion == "wolfHunt":
            world = createWolfHunt(worldSize)
        if gameVersion == "wolvesGems":
            world = createWolvesGems(worldSize)
        if gameVersion == "createGemsSearch":
            world = createGemsSearch(worldSize)

        # rewards = 0
        done = 0
        withinTurn = 0

        moveList = findMoveables(world)
        for i, j in moveList:
            world[i, j, 0].init_replay(5)

        while done == 0:

            findAgent = findAgents(world)
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

            moveList = findMoveables(world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = world[i, j, 0]

                # note the prep vision may need to be a function within the model class
                input = models[holdObject.policy].createInput(world, i, j, holdObject)

                if holdObject.static != 1:
                    # if holdObject.kind != "deadAgent":
                    action = models[holdObject.policy].takeAction([input, epsilon])

                if withinTurn == maxEpochs:
                    done = 1

            # rewrite this so all classes have transition, most are just pass
            if holdObject.has_transitions == True:
                world, models, gamePoints = holdObject.transition(
                    action,
                    world,
                    models,
                    i,
                    j,
                    gamePoints,
                    done,
                    input,
                )

            # transfer the events for each agent into the appropriate model after all have moved
            expList = findMoveables(world)
            world = updateMemories(models, world, expList, endUpdate=True)

            # below is DQN specific and we will need to come up with the general form for all models
            # but for now, write separate code for different model types to get the memory into the
            # right form for your specific model.

            # expList = findMoveables(world)
            models = transferMemories(models, world, expList)

            # testing training after every event
            if withinTurn % modelUpdate_freq == 0:
                for mods in trainableModels:
                    loss = models[mods].training(150, 0.9)
                    losses = losses + loss.detach().numpy()

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = updateEpsilon(epsilon, turn, epoch)

        # only train at the end of the game, and train each of the models that are in the model list
        # for mods in trainableModels:
        #    loss = models[mods].training(150, 0.9)
        #    losses = losses + loss.detach().numpy()

        if epoch % 100 == 0:
            print(epoch, withinTurn, gamePoints, losses, epsilon)
            gamePoints = [0, 0]
            losses = 0
    return models


def watchAgame(world, models, maxEpochs):
    fig = plt.figure()
    ims = []

    gamePoints = [0, 0]
    done = 0

    for _ in range(maxEpochs):

        image = createWorldImage(world)
        im = plt.imshow(image, animated=True)
        ims.append([im])

        moveList = findMoveables(world)
        random.shuffle(moveList)

        for i, j in moveList:
            holdObject = world[i, j, 0]

            img = agentVisualField(world, (i, j), holdObject.vision)
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            input = input.unsqueeze(0)
            if holdObject.static != 1:
                if holdObject.kind != "deadAgent":
                    action = models[holdObject.policy].takeAction([input, 0.1])

            if holdObject.has_transitions == True:
                world, models, gamePoints = holdObject.transition(
                    action,
                    world,
                    models,
                    i,
                    j,
                    gamePoints,
                    done,
                    input,
                )

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani


def createVideo(
    models, worldSize, num, gameVersion="wolfHunt", filename="unnamed_video.gif"
):
    if gameVersion == "wolfHunt":
        world = createWolfHunt(worldSize)
    if gameVersion == "wolvesGems":
        world = createWolvesGems(worldSize)
    if gameVersion == "createGemsSearch":
        world = createGemsSearch(worldSize)

    ani1 = watchAgame(world, models, 100)
    ani1.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 25, video_num, gameVersion="wolvesGems", filename=vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


# setup a game and save models (this is a quick proof of principle version that can be vastly improved on)
# note, the outputs can be better done than the hard coded print, but we need something.

# createGemsSearch
# need to update where the obejcts and videos are saved
# need to convert the games and the playgame to be classes?


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
        gameVersion="wolvesGems",  # which game to play
    )
    return models


# def train_wolf_attack(save_dir):
#     models = []
#     models.append(modelDQN(5, 0.0001, 1500, 650, 350, 100, 4))
#     models.append(modelDQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model
#     models = playGame(
#         models,  # model file list
#         [1],  # which models from that list should be trained, here not the agents
#         15,  # world size
#         25000,  # number of epochs
#         100,  # max epoch length
#         0.85,  # starting epsilon
#         gameVersion="wolfHunt",  # which game to play
#     )

#     with open(save_dir + "wolf_attack_Models_01", "wb") as fp:
#         pickle.dump(models, fp)

#     models = playGame(
#         models,  # model file list
#         [1],  # which models from that list should be trained, here not the agents
#         15,  # world size
#         50000,  # number of epochs
#         100,  # max epoch length
#         0.3,  # starting epsilon
#         gameVersion="wolfHunt",  # which game to play
#     )
#     with open(save_dir + "wolf_attack_Models_02", "wb") as fp:
#         pickle.dump(models, fp)
#     return models


# def train_gem_collector(save_dir):
#     models = []
#     models.append(modelDQN(5, 0.0001, 1500, 650, 350, 100, 4))
#     models.append(modelDQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model
#     models = playGame(
#         models,  # model file list
#         [0],  # which models from that list should be trained, here not the agents
#         15,  # world size
#         15000,  # number of epochs
#         100,  # max epoch length
#         0.85,  # starting epsilon
#         gameVersion="createGemsSearch",  # which game to play
#     )

#     with open(save_dir + "gem_collector_Models_01", "wb") as fp:
#         pickle.dump(models, fp)

#     models = playGame(
#         models,  # model file list
#         [0],  # which models from that list should be trained, here not the agents
#         15,  # world size
#         50000,  # number of epochs
#         100,  # max epoch length
#         0.3,  # starting epsilon
#         gameVersion="createGemsSearch",  # which game to play
#     )
#     with open(save_dir + "gem_collector_Models_02", "wb") as fp:
#         pickle.dump(models, fp)
#     return models


# def runCombinedTraining(
#     save_dir,
#     wolf_model,
#     agent_model,
#     trainableModels,
#     epochs,
#     max_epochs,
#     epsilon,
#     videoNum,
# ):
#     with open(save_dir + wolf_model, "rb") as fp:
#         wolf_models = pickle.load(fp)
#     with open(save_dir + agent_model, "rb") as fp:
#         agent_models = pickle.load(fp)
#     models = []
#     models.append(agent_models[0])
#     models.append(wolf_models[1])

#     models = playGame(
#         models,  # model file list
#         trainableModels,  # which models from that list should be trained, here not the agents
#         15,  # world size
#         epochs,  # number of epochs
#         max_epochs,  # max epoch length
#         epsilon,  # starting epsilon
#         gameVersion="wolvesGems",  # which game to play
#     )
#     # filename =  "GemsSearch_animation_" + str(num) + ".gif"
#     filename = save_dir + "WolfsGem_01.gif"
#     createVideo(
#         models,
#         15,
#         videoNum,
#         gameVersion="wolvesGems",
#         filename=filename,
#     )
#     with open(save_dir + "combinedModel_0001", "wb") as fp:
#         pickle.dump(models, fp)

#     return models


# def moreTraining(
#     save_dir, models, trainableModels, worldsize, epochs, max_epochs, epsilon, videoNum
# ):
#     models = playGame(
#         models,  # model file list
#         trainableModels,  # which models from that list should be trained, here not the agents
#         worldsize,  # world size
#         epochs,  # number of epochs
#         max_epochs,  # max epoch length
#         epsilon,  # starting epsilon
#         gameVersion="wolvesGems",  # which game to play
#     )
#     filename = save_dir + "WolfsGem_" + str(videoNum) + ".gif"
#     createVideo(models, 25, videoNum, gameVersion="wolvesGems", filename=filename)
#     with open("combinedModel_" + str(videoNum + 1), "wb") as fp:
#         pickle.dump(models, fp)

#     return models


# # models = runCombinedTraining(
# #    wolf_model="Model_StableWolfAttack",
# #    agent_model="modelGemsSearch_0",
# #    trainableModels=[0],
# #    epochs=30000,
# #    max_epochs=100,
# #    epsilon=0.9,
# #    videoNum=1000,
# # )


# # with open("combinedModel_1002", "rb") as fp:
# #    models = pickle.load(fp)
# # filename = "test1.gif"
# # createVideo(models, 25, 1, gameVersion="wolvesGems", filename=filename)
# # filename = "test2.gif"
# # createVideo(models, 25, 2, gameVersion="wolvesGems", filename=filename)
# # filename = "test3.gif"
# # createVideo(models, 35, 2, gameVersion="wolvesGems", filename=filename)


# # for game in range(10):
# #    models = moreTraining(models, [0], 25, 10000, 100, 0.3, game + 2000)


# # note, to really test the model, we need to build test cases. like simple
# # 9 x 9 worlds with a green and yellow gem
# # or a wolf in a location, and look at the Q values to see if avoidance is being learned

# # additional note, may need to have larger worlds to escape from wolves in


def playGameTest(
    models,
    trainableModels,
    worldSize=15,
    epochs=200000,
    maxEpochs=100,
    epsilon=0.9,
    gameVersion="wolfHunt",
):

    losses = 0
    gamePoints = [0, 0]
    status = 1
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25
    # note, rather than having random actions, we could keep the memories growing and just not train agents for
    # little while to train a different part of the model, or could stop training wolves, etc.

    for epoch in range(epochs):
        if gameVersion == "wolfHunt":
            world = createWolfHunt(worldSize)
        if gameVersion == "wolvesGems":
            world = createWolvesGems(worldSize)
        if gameVersion == "createGemsSearch":
            world = createGemsSearch(worldSize)

        # rewards = 0
        done = 0
        withinTurn = 0

        moveList = findMoveables(world)
        for i, j in moveList:
            world[i, j, 0].init_replay(5)

        while done == 0:

            findAgent = findAgents(world)
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

            moveList = findMoveables(world)
            for i, j in moveList:
                # reset the rewards for the trial to be zero for all agents
                world[i, j, 0].reward = 0
            random.shuffle(moveList)

            for i, j in moveList:
                holdObject = world[i, j, 0]

                # note the prep vision may need to be a function within the model class
                input = models[holdObject.policy].createInput(world, i, j, holdObject)

                if holdObject.static != 1:
                    # if holdObject.kind != "deadAgent":
                    action = models[holdObject.policy].takeAction([input, epsilon])

                if withinTurn == maxEpochs:
                    done = 1

            # rewrite this so all classes have transition, most are just pass
            if holdObject.has_transitions == True:
                world, models, gamePoints = holdObject.transition(
                    action,
                    world,
                    models,
                    i,
                    j,
                    gamePoints,
                    done,
                    input,
                )

            # transfer the events for each agent into the appropriate model after all have moved
            expList = findMoveables(world)
            world = updateMemories(models, world, expList, endUpdate=True)

            # below is DQN specific and we will need to come up with the general form for all models
            # but for now, write separate code for different model types to get the memory into the
            # right form for your specific model.

            # expList = findMoveables(world)
            models = transferMemories_LSTM(models, world, expList)

            # testing training after every event
            if withinTurn % modelUpdate_freq == 0:
                for mods in trainableModels:
                    loss = models[mods].training(150, 0.9)
                    losses = losses + loss.detach().numpy()

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = updateEpsilon(epsilon, turn, epoch)

        # only train at the end of the game, and train each of the models that are in the model list
        # for mods in trainableModels:
        #    loss = models[mods].training(150, 0.9)
        #    losses = losses + loss.detach().numpy()

        if epoch % 100 == 0:
            print(epoch, withinTurn, gamePoints, losses, epsilon)
            gamePoints = [0, 0]
            losses = 0
    return models, world


def train_wolf_gem_LSTM5(epochs=10000):
    models = []
    # 405 / 1445 should go back to 650 / 2570 when fixed
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 650, 350, 100, 4))  # agent model
    models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model
    models, world = playGameTest(
        models,  # model file list
        [0, 1],  # which models from that list should be trained, here not the agents
        15,  # world size
        epochs,  # number of epochs
        100,  # max epoch length
        0.85,  # starting epsilon
        gameVersion="wolvesGems",  # which game to play
    )
    return models, world
