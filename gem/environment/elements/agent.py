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
        self.just_died = False

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        image = torch.zeros(1, numberMemories, 3, 9, 9).float()
        exp = (image, 0, 0, image, 0)
        self.replay.append(exp)

    def died(
        self, models, world, attempted_locaton_1, attempted_locaton_2, extra_reward=True
    ):
        """
        Replaces the last memory with a memory that has a reward of -25 and the image of its
        death. This is to encourage the agent to not die.
        TODO: this is failing at the moment. Need to fix.
        """
        lastexp = world[attempted_locaton_1, attempted_locaton_2, 0].replay[-1]
        world[attempted_locaton_1, attempted_locaton_2, 0].replay[-1] = (
            lastexp[0],
            lastexp[1],
            -25,
            lastexp[3],
            1,
        )

        # TODO: Below is very clunky and a more principles solution needs to be found

        models[
            world[attempted_locaton_1, attempted_locaton_2, 0].policy
        ].transfer_memories(
            world, attempted_locaton_1, attempted_locaton_2, extra_reward=True
        )

        # this can only be used it seems if all agents have a different id
        self.kind = "deadAgent"  # label the agents death
        self.appearence = [130.0, 130.0, 130.0]  # dead agents are grey
        self.trainable = 0  # whether there is a network to be optimized
        self.just_died = True
        self.static = 1
        self.has_transitions = False

    def transition(
        self,
        action,
        world,
        models,
        i,
        j,
        game_points,
        done,
        input,
        update_experience_buffer=True,
        ModelType="DQN",
    ):
        """
        Sets the rules for how the agent moves and interacts with the world.
        All immplications must be written here for how the object interacts with the other objects.
        """

        new_locaton_1 = i
        new_locaton_2 = j

        # this should not be needed below, but getting errors
        # it is possible that this is fixed now with the
        # other changes that have been made
        attempted_locaton_1 = i
        attempted_locaton_2 = j

        reward = 0

        if action == 0:
            attempted_locaton_1 = i - 1
            attempted_locaton_2 = j

        if action == 1:
            attempted_locaton_1 = i + 1
            attempted_locaton_2 = j

        if action == 2:
            attempted_locaton_1 = i
            attempted_locaton_2 = j - 1

        if action == 3:
            attempted_locaton_1 = i
            attempted_locaton_2 = j + 1

        if world[attempted_locaton_1, attempted_locaton_2, 0].passable == 1:
            world[i, j, 0] = EmptyObject()
            reward = world[attempted_locaton_1, attempted_locaton_2, 0].value
            world[attempted_locaton_1, attempted_locaton_2, 0] = self
            new_locaton_1 = attempted_locaton_1
            new_locaton_2 = attempted_locaton_2
            game_points[0] = game_points[0] + reward
        else:
            if isinstance(
                world[attempted_locaton_1, attempted_locaton_2, 0], Wall
            ):  # Replacing comparison with string 'kind'
                reward = -0.1

        if update_experience_buffer == True:
            input2 = models[self.policy].pov(world, new_locaton_1, new_locaton_2, self)
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, game_points


class DeadAgent:
    """
    This is a placeholder for the dead agent. Can be replaced when .died() is corrected.
    """

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
