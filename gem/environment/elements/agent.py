from collections import deque
from gem.environment.elements.element import EmptyObject
import numpy as np
import torch
from gem.environment.elements.element import Wall


class Agent:

    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.justDied = False

    def init_replay(self, numberMemories):
        image = torch.zeros(1, numberMemories, 3, 9, 9).float()
        exp = (image, 0, 0, image, 0)
        self.replay.append(exp)

    def died(self, models, world, attLoc1, attLoc2, extraReward=True):
        lastexp = world[attLoc1, attLoc2, 0].replay[-1]
        world[attLoc1, attLoc2, 0].replay[-1] = (
            lastexp[0],
            lastexp[1],
            -25,
            lastexp[3],
            1,
        )

        # TODO: Below is very clunky and a more principles solution needs to be found

        models[world[attLoc1, attLoc2, 0].policy].transferMemories(
            world, attLoc1, attLoc2, extraReward=True
        )

        # this can only be used it seems if all agents have a different id
        # self.kind = "deadAgent"  # label the agents death
        # self.appearence = [130.0, 130.0, 130.0]  # dead agents are grey
        # self.trainable = 0  # whether there is a network to be optimized
        # self.justDied = True
        # self.static = 1
        # self.has_transitions = False

    def transition(
        self,
        action,
        world,
        models,
        i,
        j,
        gamePoints,
        done,
        input,
        expBuff=True,
        ModelType="DQN",
    ):

        newLoc1 = i
        newLoc2 = j

        # this should not be needed below, but getting errors
        # it is possible that this is fixed now with the
        # other changes that have been made
        attLoc1 = i
        attLoc2 = j

        reward = 0

        if action == 0:
            attLoc1 = i - 1
            attLoc2 = j

        if action == 1:
            attLoc1 = i + 1
            attLoc2 = j

        if action == 2:
            attLoc1 = i
            attLoc2 = j - 1

        if action == 3:
            attLoc1 = i
            attLoc2 = j + 1

        if world[attLoc1, attLoc2, 0].passable == 1:
            world[i, j, 0] = EmptyObject()
            reward = world[attLoc1, attLoc2, 0].value
            world[attLoc1, attLoc2, 0] = self
            newLoc1 = attLoc1
            newLoc2 = attLoc2
            gamePoints[0] = gamePoints[0] + reward
        else:
            if isinstance(
                world[attLoc1, attLoc2, 0], Wall
            ):  # Replacing comparison with string 'kind'
                reward = -0.1

        if expBuff == True:
            input2 = models[self.policy].pov(world, newLoc1, newLoc2, self)
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, gamePoints


class DeadAgent:

    kind = "deadAgent"  # class variable shared by all instances

    def __init__(self):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [130.0, 130.0, 130.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = "NA"  # agent model here.
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 1  # whether the object gets to take actions or not (starts as 0, then goes to 1)
        self.passable = 0  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.replay = deque([], maxlen=5)
        self.has_transitions = False
