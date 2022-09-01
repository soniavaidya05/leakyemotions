#!/usr/bin/env python
# coding: utf-8

from old_files.utils import (
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
    StaticAgent,
    Wolf,
    Gem,
    Wall,
    deadAgent,
)
from old_game_file.game_utils import createWorld, createWorldImage
from old_game_file.transitions import agentTransitions, wolfTransitions


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
staticAgent1 = StaticAgent(0)

# create the instances
def createWolfHunt(
    worldSize, staticAgents=False, gem1p=0.115, gem2p=0.06, agent1p=0.05
):

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
                if staticAgents == False:
                    world[i, j, 0] = agent1
                if staticAgents == True:
                    world[i, j, 0] = gem3

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


def createWolvesGems(
    worldSize, staticAgents=False, gem1p=0.115, gem2p=0.06, agent1p=0.005
):

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
                if staticAgents == False:
                    world[i, j, 0] = wolf1
                if staticAgents == True:
                    world[i, j, 0] = gem3

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


def createGemsSearch(
    worldSize, staticAgents=False, gem1p=0.115, gem2p=0.06, agent1p=0.00
):

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
                if staticAgents == False:
                    world[i, j, 0] = wolf1
                if staticAgents == True:
                    world[i, j, 0] = gem3

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


def playGame(
    models,
    trainableModels,
    worldSize=15,
    epochs=200000,
    maxEpochs=100,
    epsilon=0.9,
    staticAgents=False,
    gameVersion="wolfHunt",
):

    losses = 0
    totalRewards = 0
    status = 1
    turn = 0
    sync_freq = 500
    modelUpdate_freq = 25
    # note, rather than having random actions, we could keep the memories growing and just not train agents for
    # little while to train a different part of the model, or could stop training wolves, etc.

    for epoch in range(epochs):
        if gameVersion == "wolfHunt":
            world = createWolfHunt(worldSize, staticAgents)
        if gameVersion == "wolvesGems":
            world = createWolvesGems(worldSize, staticAgents)
        if gameVersion == "createGemsSearch":
            world = createGemsSearch(worldSize, staticAgents)

        rewards = 0
        done = 0
        withinTurn = 0
        wolfEats = 0
        agentEats = 0
        while done == 0:

            if staticAgents == False:
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

                img = agentVisualField(world, (i, j), holdObject.vision)
                input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
                if staticAgents == False:
                    if holdObject.static != 1:
                        if holdObject.kind != "deadAgent":
                            action = models[holdObject.policy].takeAction(
                                [input, epsilon]
                            )
                if staticAgents == True:
                    if holdObject.kind != "agent":
                        if holdObject.static != 1:
                            if holdObject.kind != "deadAgent":
                                action = models[holdObject.policy].takeAction(
                                    [input, epsilon]
                                )

                if withinTurn == maxEpochs:
                    done = 1

                if holdObject.kind == "agent":
                    world, models, totalRewards = agentTransitions(
                        holdObject,
                        action,
                        world,
                        models,
                        i,
                        j,
                        totalRewards,
                        done,
                        input,
                    )

                if holdObject.kind == "wolf":
                    world, models, wolfEats = wolfTransitions(
                        holdObject, action, world, models, i, j, wolfEats, done, input
                    )

            # transfer the events for each agent into the appropriate model after all have moved
            expList = findMoveables(world)
            world = updateMemories(world, expList, endUpdate=True)

            # below is DQN specific and we will need to come up with the general form for all models
            # but for now, write separate code for different model types to get the memory into the
            # right form for your specific model.

            expList = findMoveables(world)
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
            print(epoch, withinTurn, totalRewards, wolfEats, losses, epsilon)
            wolfEats = 0
            agentEats = 0
            losses = 0
            totalRewards = 0
    return models


def watchAgame(world, models, maxEpochs):
    fig = plt.figure()
    ims = []

    totalRewards = 0
    wolfEats = 0
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
            if holdObject.static != 1:
                if holdObject.kind != "deadAgent":
                    action = models[holdObject.policy].takeAction([input, 0.1])

            if holdObject.kind == "agent":
                world, models, totalRewards = agentTransitions(
                    holdObject,
                    action,
                    world,
                    models,
                    i,
                    j,
                    totalRewards,
                    done,
                    input,
                )

            if holdObject.kind == "wolf":
                world, models, wolfEats = wolfTransitions(
                    holdObject, action, world, models, i, j, wolfEats, done, input
                )

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani


def createVideo(worldSize, num, gameVersion="wolfHunt"):
    filename = "GemsSearch_animation_" + str(num) + ".gif"
    if gameVersion == "wolfHunt":
        world = createWolfHunt(worldSize, staticAgents=False)
    if gameVersion == "wolvesGems":
        world = createWolvesGems(worldSize, staticAgents=False)
    ani1 = watchAgame(world, models, 100)
    ani1.save(filename, writer="PillowWriter", fps=2)


# setup a game and save models (this is a quick proof of principle version that can be vastly improved on)
# note, the outputs can be better done than the hard coded print, but we need something.

newModels = 2

# create neuralnet models
if newModels == 1:
    models = []
    models.append(modelRandomAction(10, 4))  # agent1 model
    models.append(modelDQN(5, 0.0001, 650, 2570, 350, 100, 4))
    models.append(modelDQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model

    for games in range(2):
        models = playGame(
            models,
            [1],
            15,
            10000,
            100,
            0.85,
            staticAgents=False,
            gameVersion="wolfHunt",
        )
        with open("modelGemsSearch_" + str(games), "wb") as fp:
            pickle.dump(models, fp)
        createVideo(15, games, gameVersion="createGemsSearch")

if newModels == 2:
    with open("modelGemsSearch_10", "rb") as fp:
        models = pickle.load(fp)


# let the agents start to learn the world as well and move to a larger world
for games in range(1):
    models[0] = modelDQN(5, 0.0001, 1500, 650, 350, 100, 4)
    models = playGame(
        models,
        [0, 1],
        25,
        10000,
        100,
        0.95,
        staticAgents=False,
        gameVersion="wolvesGems",
    )
    with open("modelGemsSearch_" + str(games + 10), "wb") as fp:
        pickle.dump(models, fp)
    createVideo(25, games + 10, gameVersion="createGemsSearch")

for games in range(5):
    models = playGame(
        models,
        [0, 1],
        15,
        10000,
        100,
        0.3,
        staticAgents=False,
        gameVersion="wolvesGems",
    )
    with open("modelGemsSearch_" + str(games + 10), "wb") as fp:
        pickle.dump(models, fp)
    createVideo(15, games + 20, gameVersion="createGemsSearch")

# createGemsSearch
