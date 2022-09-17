from gem.environment.elements.element import Element
import random
from collections import deque
import numpy as np
import torch
from gem.environment.elements.element import Wall


class Wood(Element):

    kind = "wood"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the wood, whether it has been mined or not
        self.appearence = (160, 82, 45)  # wood is brown
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 0  # the value of this wood
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False


class Stone(Element):

    kind = "stone"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearence = (136, 140, 141)  # stone is grey
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False


class House(Element):

    kind = "house"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearence = (220, 216, 199)  # houses are tan
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False


class EmptyObject(Element):

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # empty stuff is basically empty
        self.appearence = [0.0, 0.0, 0.0]  # empty is well, blank
        self.vision = 1  # empty stuff is basically empty
        self.policy = "NA"  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.reward = 0  # empty stuff is basically empty
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.stone = 0
        self.wood = 0

    def transition(self, world, models, action, location):
        generate_value = np.random.choice([0, 1, 2], p=[0.95, 0.025, 0.025])
        if generate_value == 1:
            world[location] = Wood()
        if generate_value == 2:
            world[location] = Stone()

        reward = 0
        next_state = 0
        done = 0
        new_loc = location
        return world, reward, next_state, done, new_loc


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
        self.stone = 0
        self.wood = 0

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 9
        image = torch.zeros(1, numberMemories, 3, pov_size, pov_size).float()
        exp = (image, 0, 0, image, 0)
        self.replay.append(exp)

    def transition(self, world, models, action, location):
        i, j, k = location
        done = 0
        new_loc = location

        attempted_locaton_1 = i
        attempted_locaton_2 = j
        attempted_locaton_3 = k

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

        attempted_location = (
            attempted_locaton_1,
            attempted_locaton_2,
            attempted_locaton_3,
        )

        if action < 4:
            if isinstance(world[attempted_location], Wall):
                reward = -0.1

            if isinstance(world[attempted_location], House):
                reward = -0.1

            if isinstance(world[attempted_location], Wood):
                reward = 1
                self.wood += 1
                world[attempted_location] = self
                world[location] = EmptyObject()
                new_loc = attempted_location

            if isinstance(world[attempted_location], Stone):
                reward = 1
                self.stone += 1
                world[attempted_location] = self
                world[location] = EmptyObject()
                new_loc = attempted_location

        if action == 5:
            move_to = np.array([1, 1, 1, 1])
            if world[i + 1, j, k] != EmptyObject():
                move_to[0] = 0
            if world[i - 1, j, k] != EmptyObject():
                move_to[1] = 0
            if world[i, j - 1, k] != EmptyObject():
                move_to[2] = 0
            if world[i, j + 1, k] != EmptyObject():
                move_to[3] = 0

            move_to = move_to / np.sum(move_to)
            movement = np.random.choice([0, 1, 2, 3], p=move_to)

            if movement == 0:
                attempted_locaton_1 = i + 1
                attempted_locaton_2 = j
            if movement == 1:
                attempted_locaton_1 = i - 1
                attempted_locaton_2 = j
            if movement == 2:
                attempted_locaton_1 = i
                attempted_locaton_2 = j - 1
            if movement == 3:
                attempted_locaton_1 = i
                attempted_locaton_2 = j + 1
            attempted_location = (
                attempted_locaton_1,
                attempted_locaton_2,
                attempted_locaton_3,
            )
            if self.stone > 0 and self.wood > 0 and np.sum(move_to) > 0:
                print("built a house!!")
                reward = 10
                self.stone -= 1
                self.wood -= 1
                world[attempted_location] = self
                world[location] = EmptyObject()
                new_loc = attempted_location

        next_state = models[self.policy].pov(world, new_loc, self)
        self.reward += reward

        return world, reward, next_state, done, new_loc
