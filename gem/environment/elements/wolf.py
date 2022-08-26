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
        img = np.random.rand(17, 17, 3) * 0
        state = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        if numberMemories > 0:
            state = state.unsqueeze(0)
        exp = (state, 0, 0, state, 0)
        self.replay.append(exp)
        if numberMemories > 0:
            for _ in range(numberMemories):
                self.replay.append(exp)

    def transition(
        self, action, world, models, i, j, gamePoints, done, input, expBuff=True
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

                # update the last memory of the agent that was eaten

                lastexp = world[attLoc1, attLoc2, 0].replay[-1]
                world[attLoc1, attLoc2, 0].replay[-1] = (
                    lastexp[0],
                    lastexp[1],
                    -25,
                    lastexp[3],
                    1,
                )
                models[world[attLoc1, attLoc2, 0].policy].transferMemories(
                    world, attLoc1, attLoc2, extraReward=True
                )

                world[attLoc1, attLoc2, 0] = DeadAgent()

        if expBuff == True:
            input2 = models[self.policy].createInput(world, newLoc1, newLoc1, self)
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, gamePoints