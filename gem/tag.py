#!/usr/bin/env python
# coding: utf-8

from gem.utils import (
    one_hot,
    updateEpsilon,
    updateMemories,
    transferMemories,
    findMoveables,
    findAgents_tag,
)


# replay memory class

from models.memory import Memory
from models.dqn import DQN, modelDQN
from models.perception import agentVisualField
from environment.elements import TagAgent, EmptyObject, Wall
from gem.game_utils import createWorld, createWorldImage
from tagworld.transitions import tagAgentTransitions


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


def createTagWorld(worldSize, agentp=.05):
    num_agents = 0
    emptyObject = EmptyObject()
    walls = Wall()

    # make the world and populate
    world = createWorld(worldSize, worldSize, 1, emptyObject)
    for i in range(worldSize):
        for j in range(worldSize):
            obj = np.random.choice([0, 1], p=[agentp, 1 - agentp])
            if obj == 0:
                world[i, j, 0] = TagAgent(0)
                num_agents += 1
    
        #world[round(worldSize/2),round(worldSize/2),0] = agent1
        #world[round((worldSize/2))-1,(round(worldSize/2))+1,0] = agent2
    for i in range(worldSize):
        world[0, i, 0] = walls
        world[worldSize-1, i, 0] = walls
        world[i, 0, 0] = walls
        world[i, worldSize-1, 0] = walls
    
    agents = findAgents_tag(world)
    itAgent = random.choice(agents)
    itAgent.tag()

    # need to initialize replay buffer of each agent
    # state and next state - curstate
    # action = random(0,4)
    # reward = 0
    # done = 0
    all_agents = findMoveables(world)
    for i, j in all_agents:
        agent = world[i,j,0]
        img = agentVisualField(world, (i, j), agent.vision)
        current_state = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        next_state = current_state
        action = random.randint(0,3)
        reward = 0
        done = 0
        exp = (current_state, action, reward, next_state, done)
        agent.replay.append(exp)

    return world

# test the world models

def plotWorld(world):
    img = createWorldImage(world)
    plt.imshow(img)
    plt.show()

def gameTest(worldSize):
    world = createTagWorld(worldSize)
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
# world1 = createTagWorld(30)
# plt.imshow(createWorldImage(world1))
# plt.show()

def playGame(models, worldSize=15, epochs=200000, maxEpochs=100, epsilon=0.9):

    losses = 0
    totalRewards = 0
    status = 1
    turn = 0
    sync_freq = 500

    for epoch in range(epochs):
        # print(epoch)
        world = createTagWorld(worldSize)
        done = 0
        withinTurn = 0
        while done == 0:

            withinTurn = withinTurn + 1
            turn = turn + 1

            if turn % sync_freq == 0:
                for mods in range(len(models)):
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
                if holdObject.static != 1:
                    action = models[holdObject.policy].takeAction([input, epsilon])

                if withinTurn == maxEpochs:
                    done = 1

                if holdObject.kind == "TagAgent":
                    world, models, totalRewards = tagAgentTransitions(
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

            # transfer the events for each agent into the appropriate model after all have moved
            expList = findMoveables(world)
            world = updateMemories(world, expList, endUpdate=True)

            # below is DQN specific and we will need to come up with the general form for all models
            # but for now, write separate code for different model types to get the memory into the
            # right form for your specific model.

            expList = findMoveables(world)
            models = transferMemories(models, world, expList)

        # epdate epsilon to move from mostly random to greedy choices for action with time
        epsilon = updateEpsilon(epsilon, turn, epoch)

        # only train at the end of the game, and train each of the models that are in the model list
        for mod in range(len(models)):
            loss = models[mod].training(150, 0.9)
            losses = losses + loss.detach().numpy()

        if epoch % 100 == 0:
            print(epoch, totalRewards, losses, epsilon)
            losses = 0
            totalRewards = 0
    return models

tmp = playGame(models, 15, 1000, 100, .85)
models = playGame(models, 15, 10000, 100, 0.85)

def watchAgame(world, models, maxEpochs):
    fig = plt.figure()
    ims = []

    totalRewards = 0
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
                action = models[holdObject.policy].takeAction([input, 0.1])

            if holdObject.kind == "TagAgent":
                world, models, totalRewards = tagAgentTransitions(
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

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani

world3 = createTagWorld(30)
video = watchAgame(world3, models, maxEpochs=200)
video.save('./tag_output/tag_vid_1.gif', writer="PillowWriter", fps=2)

def createVideo(worldSize, num):
    filename = "animation_A" + str(num) + ".gif"
    world = createGemWorld(worldSize)
    ani1 = watchAgame(world, models, 100)
    ani1.save(filename, writer="PillowWriter", fps=2)

    filename = "animation_B" + str(num) + ".gif"
    world = createGemWorldTest(worldSize)
    ani2 = watchAgame(world, models, 100)
    ani2.save(filename, writer="PillowWriter", fps=2)


# setup a game and save models (this is a quick proof of principle version that can be vastly improved on)
# note, the outputs can be better done than the hard coded print, but we need something.

newModels = 1

# create neuralnet models
if newModels == 1:
    models = []
    models.append(modelDQN(5, 0.0001, 1500, 650, 350, 100, 4))  # agent1 model
    models.append(modelDQN(5, 0.0001, 1500, 650, 350, 100, 4))  # agent2 model
    models = playGame(models, 15, 10000, 100, 0.85)
    with open("tagModelFile", "wb") as fp:
        pickle.dump(models, fp)
    createVideo(30, 0)


if newModels == 2:
    with open("modelFile", "rb") as fp:
        models = pickle.load(fp)

for games in range(20):
    models = playGame(models, 15, 10000, 100, 0.3)
    with open("modelFile_" + str(games), "wb") as fp:
        pickle.dump(models, fp)
    createVideo(30, games + 1)
