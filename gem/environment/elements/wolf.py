from collections import deque
from gem.environment.elements.element import EmptyObject
from gem.environment.elements.element import Wall
from gem.environment.elements.agent import Agent, DeadAgent
import numpy as np
import torch


class Wolf:

    kind = "wolf"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearence = [255.0, 0.0, 0.0]  # agents are red
        self.vision = 8  # agents can see three radius around them
        self.policy = model  # gems do not do anything
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=5)  # we should read in these maxlens
        self.has_transitions = True

    # init is now for LSTM, may need to have a toggle for LSTM of not
    def init_replay(self, numberMemories):
        image = torch.zeros(1, numberMemories, 3, 17, 17).float()
        exp = (image, 0, 0, image, 0)
        self.replay.append(exp)

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
            if world[attLoc1, attLoc2, 0].appearence == [0.0, 0.0, 255.0]:
                reward = 10
                wolfEats = wolfEats + 1
            world[i, j, 0] = EmptyObject()
            world[attLoc1, attLoc2, 0] = self
            newLoc1 = attLoc1
            newLoc2 = attLoc2
            reward = 0
        else:
            if isinstance(world[attLoc1, attLoc2, 0], Wall):
                reward = -0.1
            if isinstance(world[attLoc1, attLoc2, 0], Agent):
                reward = 10
                gamePoints[1] = gamePoints[1] + 1
                newVersion = 0
                if newVersion == 0:

                    # update the last memory of the agent that was eaten

                    lastexp = world[attLoc1, attLoc2, 0].replay[-1]
                    world[attLoc1, attLoc2, 0].replay[-1] = (
                        lastexp[0],
                        lastexp[1],
                        -25,
                        lastexp[3],
                        1,
                    )

                    # TODO: Below is very clunky and a more principles solution needs to be found

                    if ModelType == "DQN":
                        models[world[attLoc1, attLoc2, 0].policy].transferMemories(
                            world, attLoc1, attLoc2, extraReward=True
                        )
                    if ModelType == "AC":
                        # note, put in the whole code for updatng an AC model here
                        if len(world[attLoc1, attLoc2, 0].AC_value) > 0:

                            finalReward = torch.tensor(-25).float().reshape(1, 1)

                            if (
                                world[attLoc1, attLoc2, 0].AC_reward.shape
                                == world[attLoc1, attLoc2, 0].AC_value.shape
                            ):
                                world[attLoc1, attLoc2, 0].AC_reward[-1] = finalReward
                            else:
                                world[attLoc1, attLoc2, 0].AC_reward = torch.concat(
                                    [world[attLoc1, attLoc2, 0].AC_reward, finalReward]
                                )
                                models[
                                    world[attLoc1, attLoc2, 0].policy
                                ].transferMemories_AC(world, attLoc1, attLoc2)

                    world[attLoc1, attLoc2, 0] = DeadAgent()
                if newVersion == 1:
                    world = world[attLoc1, attLoc2, 0].died(
                        models, world, attLoc1, attLoc2, extraReward=True
                    )
                    world[attLoc1, attLoc2, 0] = DeadAgent()

        if expBuff == True:
            input2 = models[self.policy].pov(world, newLoc1, newLoc2, self)
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, gamePoints
