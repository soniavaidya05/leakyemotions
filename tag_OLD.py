import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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

import objectClasses


def findAgents(world):
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == 'agent':
                agentList.append(world[i, j, 0])
    return agentList

def createTagWorldMulti(worldSize, agentp=.05):
    agents = 0

    # make the world and populate
    world = createWorld(worldSize, worldSize, 1, emptyObject)
    for i in range(worldSize):
        for j in range(worldSize):
            obj = np.random.choice([0, 1], p=[agentp, 1 - agentp])
            if obj == 0:
                world[i, j, 0] = tagAgent(0)
                agents = + 1
    
    agents = findAgents(world)
    itAgent = random.choice(agents)
    itAgent.tag()

        #world[round(worldSize/2),round(worldSize/2),0] = agent1
        #world[round((worldSize/2))-1,(round(worldSize/2))+1,0] = agent2
    for i in range(worldSize):
        world[0, i, 0] = walls
        world[worldSize-1, i, 0] = walls
        world[i, 0, 0] = walls
        world[i, worldSize-1, 0] = walls

    return world


# ---------------------------------------------------------------------
#                      Agent transition rules (updated)
# ---------------------------------------------------------------------
def tagAgentTransitions(holdObject, action, world, models, i, j, rewards, totalRewards, done, input, expBuff=True):

    newLoc1 = i
    newLoc2 = j

    reward = 0

    # if object is frozen, just decrease frozen count and then go from there
    # hope this doesn't break learning
    if holdObject.frozen > 0:
        holdObject.frozen -= 1
        return world, models, totalRewards

    if action == 0:
        attLoc1 = i-1
        attLoc2 = j
    elif action == 1:
        attLoc1 = i+1
        attLoc2 = j
    elif action == 2:
        attLoc1 = i
        attLoc2 = j-1
    elif action == 3:
        attLoc1 = i
        attLoc2 = j+1

    if world[attLoc1, attLoc2, 0].passable == 1:
        world[i, j, 0] = EmptyObject()
        reward += 1
        world[attLoc1, attLoc2, 0] = holdObject
        newLoc1 = attLoc1
        newLoc2 = attLoc2
        totalRewards += 1
    elif world[attLoc1, attLoc2, 0].kind == "tagAgent":
        # bump into each other
        alterAgent = world[attLoc1, attLoc2, 0]
        if holdObject.is_it == 1 & alterAgent.is_it == 0:
            reward += 7
            holdObject.tag()  # change who is it
            alterAgent.tag()

            if expBuff == True:

                imgDead = agentVisualField(
                    world, (attLoc1, attLoc2), alterAgent.vision)
                inputDead = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                QDead = models[world[attLoc1, attLoc2, 0].policy].model1(
                    inputDead)
                pDead = m(QDead).detach().numpy()[0]
                actionDead = np.random.choice([0, 1, 2, 3], p=pDead)

                imgDead2 = agentVisualField(
                    world, (attLoc1, attLoc2, j), world[attLoc1, attLoc2, 0].vision)
                inputDead2 = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                exp = (inputDead, actionDead, -10, inputDead2, 1)
                models[world[attLoc1, attLoc2, 0].policy].replay.append(exp)
        elif holdObject.is_it == 0 & alterAgent.is_it == 1:
            # agent bumpbed into tag person
            reward -= 13
            holdObject.tag()  # change who is it
            alterAgent.tag()

            if expBuff == True:

                imgDead = agentVisualField(
                    world, (attLoc1, attLoc2), alterAgent.vision)
                inputDead = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                QDead = models[world[attLoc1, attLoc2, 0].policy].model1(
                    inputDead)
                pDead = m(QDead).detach().numpy()[0]
                actionDead = np.random.choice([0, 1, 2, 3], p=pDead)

                imgDead2 = agentVisualField(
                    world, (attLoc1, attLoc2, j), world[attLoc1, attLoc2, 0].vision)
                inputDead2 = torch.tensor(imgDead).unsqueeze(
                    0).permute(0, 3, 1, 2).float()
                exp = (inputDead, actionDead, 7, inputDead2, 1)
                models[world[attLoc1, attLoc2, 0].policy].replay.append(exp)
        elif holdObject.is_it == alterAgent.is_it:
            # bump into each other
            reward -= .1

    elif world[attLoc1, attLoc2, 0].kind == "wall":
        reward = -.1

    if expBuff == True:
        img2 = agentVisualField(world, (newLoc1, newLoc2), holdObject.vision)
        input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
        exp = (input, action, reward, input2, done)
        models[holdObject.policy].replay.append(exp)

    return world, models, totalRewards

