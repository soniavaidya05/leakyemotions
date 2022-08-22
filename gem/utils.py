import numpy as np
from models.perception import agentVisualField
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def findMoveables(world):
    moveList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].static == 0:
                moveList.append([i, j])
    return moveList


# we will want to have a single "find" script that takes as input what you are looking for and finds those objects
def findAgents_tag(world):
    print(
        "the findAgents_tag function will be deleted soon. Please update to findInstance"
    )
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == "TagAgent":
                agentList.append(world[i, j, 0])
    return agentList


def findAgents(world):
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == "agent":
                agentList.append(world[i, j, 0])
    return agentList


def findInstance(world, kind):
    instList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == kind:
                instList.append(world[i, j, 0])
    return instList


def updateEpsilon(epsilon, turn, epoch):
    if epsilon > 0.1:
        epsilon -= 1 / (turn)

    if epsilon > 0.2:
        if epoch > 1000 and epoch % 10000 == 0:
            epsilon -= 0.1
    # print("new epsilon", epsilon)
    return epsilon


# view a replay memory


def examineReplay(models, index, modelnum):

    image_r = models[modelnum].replay[index][0][0][0].numpy()
    image_g = models[modelnum].replay[index][0][0][1].numpy()
    image_b = models[modelnum].replay[index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay[index][3][0][0].numpy()
    image_g = models[modelnum].replay[index][3][0][1].numpy()
    image_b = models[modelnum].replay[index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay[index][1]
    reward = models[modelnum].replay[index][2]
    done = models[modelnum].replay[index][4]

    return image, image2, (action, reward, done)


# look at a few replay games
def replayGames(numGames, modelNum, startingMem, models):
    for i in range(numGames):
        image, image2, memInfo = examineReplay(models, i + startingMem, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")

        print(memInfo)

        plt.show()


def examineReplayMemory(models, episode, index, modelnum):

    image_r = models[modelnum].replay.memory[episode][index][0][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][0][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay.memory[episode][index][3][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][3][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay.memory[episode][index][1]
    reward = models[modelnum].replay.memory[episode][index][2]
    done = models[modelnum].replay.memory[episode][index][4]

    return image, image2, (action, reward, done)


# look at a few replay games


def replayMemoryGames(models, modelNum, episode):
    epLength = len(models[modelNum].replay.memory[episode])
    for i in range(epLength):
        image, image2, memInfo = examineReplayMemory(models, episode, i, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")

        print(memInfo)

        plt.show()


def numberOfMemories(modelNum, models):
    episodes = len(models[modelNum].replay.memory)
    print("there are ", episodes, " in the model replay buffer.")
    for e in range(episodes):
        epLength = len(models[0].replay.memory[e])
        print("Memory ", e, " is ", epLength, " long.")


def updateMemories(world, expList, endUpdate=True):
    # update the reward and last state after all have moved
    for i, j in expList:
        # note, should these replay[0]s be replay[-1] in case we need to store more memories?
        exp = world[i, j, 0].replay[-1]
        if endUpdate == True:
            img2 = agentVisualField(world, (i, j), world[i, j, 0].vision)
            input2 = torch.tensor(img2).unsqueeze(0).permute(0, 3, 1, 2).float()
            exp = (exp[0], exp[1], world[i, j, 0].reward, input2, exp[4])
        world[i, j, 0].replay[0] = exp
    return world


def transferMemories(models, world, expList, extraReward=True):
    # transfer the events from agent memory to model replay
    for i, j in expList:
        # note, should these replay[0]s be replay[-1] in case we need to store more memories?
        exp = world[i, j, 0].replay[-1]
        models[world[i, j, 0].policy].replay.append(exp)
        if extraReward == True and abs(exp[2]) > 9:
            for _ in range(5):
                models[world[i, j, 0].policy].replay.append(exp)
    return models
