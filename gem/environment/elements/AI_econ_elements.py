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
        self.appearence = (60, 225, 45)  # wood is brown
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 0  # the value of this wood
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic


class Stone(Element):

    kind = "stone"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearence = (116, 120, 121)  # stone is grey
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic


class House(Element):

    kind = "house"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearence = (225, 0, 0)  # houses are red
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic


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
        self.deterministic = 1  # whether the object is deterministic
        self.has_transitions = False
        self.stone = 0
        self.wood = 0

    def transition(self, world, location):
        generate_value = np.random.choice([0, 1, 2], p=[0.9999, 0.00005, 0.00005])
        if generate_value == 1:
            world[location] = Wood()
        if generate_value == 2:
            world[location] = Stone()
        return world


class Agent:

    # whole thing needs to be reconsidered because element interactions
    # are in different layers

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
        self.deterministic = 0  # whether the object is deterministic
        self.labour = 0

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 9
        visual_depth = 3 + 3 + 3
        image = torch.zeros(1, numberMemories, visual_depth, pov_size, pov_size).float()
        exp = (image, 0, 0, image, 0)
        self.replay.append(exp)

    def transition(self, world, models, action, location):
        i, j, k = location
        done = 0
        new_loc = location

        # given that motion is in nearly all models, we should have a function
        #   that computes location and attempted location

        attempted_locaton_1 = i
        attempted_locaton_2 = j
        attempted_locaton_3 = k

        reward = 0

        if action == 0:
            attempted_locaton_1 = i - 1
            attempted_locaton_2 = j
            self.labour -= 0.21

        if action == 1:
            attempted_locaton_1 = i + 1
            attempted_locaton_2 = j
            self.labour -= 0.21

        if action == 2:
            attempted_locaton_1 = i
            attempted_locaton_2 = j - 1
            self.labour -= 0.21

        if action == 3:
            attempted_locaton_1 = i
            attempted_locaton_2 = j + 1
            self.labour -= 0.21

        attempted_location_l0 = (
            attempted_locaton_1,
            attempted_locaton_2,
            0,
        )

        attempted_location_l1 = (
            attempted_locaton_1,
            attempted_locaton_2,
            1,
        )

        if action < 3.5:
            self.labour -= 2.1
            if isinstance(world[attempted_location_l1], Agent):
                reward = -0.1

            else:
                if isinstance(world[attempted_location_l0], EmptyObject):
                    world[attempted_location_l1] = self
                    new_loc = attempted_location_l1
                    world[location] = EmptyObject()

                if isinstance(world[attempted_location_l0], Wall):
                    reward = -0.1

                if isinstance(world[attempted_location_l0], House):
                    reward = -0.1

                if isinstance(world[attempted_location_l0], Wood):
                    # once this works, we need to set the reward to be 0 for collecting
                    # labour costs need to be implimented
                    reward = 10
                    self.wood += 1
                    world[attempted_location_l1] = self
                    world[attempted_location_l0] = EmptyObject()
                    world[location] = EmptyObject()
                    new_loc = attempted_location_l1

                if isinstance(world[attempted_location_l0], Stone):
                    reward = 10
                    self.stone += 1
                    world[attempted_location_l1] = self
                    world[attempted_location_l0] = EmptyObject()
                    world[location] = EmptyObject()
                    new_loc = attempted_location_l1

        if action == 4:
            # this is a hack and needs to be better conceptualized
            # solution is likely to have the agents and the elements (wood, stone, houses)
            #   be on two different layers. So an agent and a house can be on
            #   the same i, j, rather than having to have the agent jump off

            # note, you should not be able to build on top of another house
            reward = -0.1
            if self.stone > 0 and self.wood > 0:
                if isinstance(world[location[0], location[1], 0], EmptyObject):
                    reward = 100
                    self.stone -= 1
                    self.wood -= 1
                    world[location[0], location[1], 0] = House()
                    succeed_house = 1

        next_state = models[self.policy].pov(
            world,
            new_loc,
            self,
            inventory=[self.stone, self.wood, self.labour],
            layers=[0, 1],
        )
        self.reward += reward

        return world, reward, next_state, done, new_loc


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # wall stuff is basically empty
        self.appearence = [153.0, 51.0, 102.0]  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
