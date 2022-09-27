import numpy as np
from models.perception import agent_visualfield
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def find_moveables(world):
    # needs to be rewriien to return location (i, j, k)
    moveList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].static == 0:
                    moveList.append((i, j, k))
    return moveList


def find_trainables(world):
    # needs to be rewriien to return location (i, j, k)
    trainList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].trainable == 1:
                    trainList.append((i, j, k))
    return trainList


# we will want to have a single "find" script that takes as input what you are looking for and finds those objects
def find_agents_tag(world):
    # needs to be rewriien to return location (i, j, k)
    print(
        "the find_agents_tag function will be deleted soon. Please update to find_instance"
    )
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == "TagAgent":
                agentList.append((i, j, 0))
    return agentList


def find_agents(world):
    # needs to be rewriien to return location (i, j, k)
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].kind == "agent":
                    agentList.append((i, j, k))
    return agentList


def find_instance(world, kind):
    # needs to be rewriien to return location (i, j, k)
    instList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i, j, 0].kind == kind:
                instList.append((i, j, 0))
    return instList


def update_epsilon(epsilon, rate=0.99):
    # TODO, this goes down way too fast when turn is low
    # so, fast updatee mean that epsilon goes down faster than it should
    # if epsilon > 0.1:
    #    epsilon -= 1 / (turn)

    epsilon = max(0.1, rate * epsilon)

    return epsilon


# view a replay memory


def examine_replay(models, index, modelnum):

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
def replay_games(numGames, modelNum, startingMem, models):
    for i in range(numGames):
        image, image2, memInfo = examine_replay(models, i + startingMem, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")

        print(memInfo)

        plt.show()


def examine_replay_memory(models, episode, index, modelnum):

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


def replay_MemoryGames(models, modelNum, episode):
    epLength = len(models[modelNum].replay.memory[episode])
    for i in range(epLength):
        image, image2, memInfo = examine_replay_memory(models, episode, i, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")
        print(memInfo)
        plt.show()


def number_memories(modelNum, models):
    episodes = len(models[modelNum].replay.memory)
    print("there are ", episodes, " in the model replay buffer.")
    for e in range(episodes):
        epLength = len(models[0].replay.memory[e])
        print("Memory ", e, " is ", epLength, " long.")


def update_memories(models, world, expList, done, end_update=True):
    # update the reward and last state after all have moved
    # changed to holdObject to see if this fixes the failure of updating last memory
    for loc in expList:
        # location = (i, j, 0)
        holdObject = world[loc]
        exp = holdObject.replay[-1]
        lastdone = exp[1][4]
        if done == 1:
            lastdone = 1
        if end_update == False:
            exp = exp[0], (exp[1][0], exp[1][1], holdObject.reward, exp[1][3], lastdone)
        if end_update == True:
            input2 = models[holdObject.policy].pov(world, loc, holdObject)
            exp = exp[0], (exp[1][0], exp[1][1], holdObject.reward, input2, lastdone)
        world[loc].replay[-1] = exp
    return world


def transfer_world_memories(models, world, expList, extra_reward=True):
    # transfer the events from agent memory to model replay
    for loc in expList:
        # this moves the specific form of the replay memory into the model class
        # where it can be setup exactly for the model
        models[world[loc].policy].transfer_memories(world, loc, extra_reward=True)
    return models


def transfer_memories(models, world, expList, extra_reward=True):
    # transfer the events from agent memory to model replay
    for loc in expList:
        exp = world[loc].replay[-1]
        models[world[loc].policy].replay.append(exp)
        if extra_reward == True and abs(exp[1][2]) > 3:
            for _ in range(2):
                models[world[loc].policy].replay.append(exp)
    return models
