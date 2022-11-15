import random
from collections import deque
import numpy as np
import torch

class Wood():

    kind = "wood"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the wood, whether it has been mined or not
        self.appearance = (60, 225, 45)  # wood is brown
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 0  # the value of this wood
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"


class Stone():

    kind = "stone"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearance = (116, 120, 121)  # stone is grey
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"


class House():

    kind = "house"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.health = 1  # for the stone, whether it has been mined or not
        self.appearance = (225, 0, 0)  # houses are red
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = 10  # the value of this stone
        self.reward = 0  # how much reward this wood has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"


class EmptyObject():

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # empty stuff is basically empty
        self.appearance = [0.0, 0.0, 0.0]  # empty is well, blank
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
        self.action_type = "deterministic"

    def transition(self, world, location):
        generate_value = np.random.choice([0, 1, 2], p=[0.9999, 0.00005, 0.00005])
        if generate_value == 1:
            world[location] = Wood()
        if generate_value == 2:
            world[location] = Stone()
        return world


class Agent:

    kind = "agent"  # class variable shared by all instances

    def __init__(self, model, wood_skill, stone_skill, house_skill, appearance):
        self.appearance = appearance  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.deterministic = 0  # whether the object is deterministic
        self.stone = 0
        self.wood = 0
        self.labour = 0
        self.action_type = "neural_network"
        self.wood_skill = wood_skill
        self.stone_skill = stone_skill
        self.house_skill = house_skill
        self.coin = 0
        self.init_rnn_state = None


    def init_replay(self, numberMemories, pov_size = 9, visual_depth = 9):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        pov_size = 9
        visual_depth = 3 + 3 + 3
        image = torch.zeros(1, numberMemories, visual_depth, pov_size, pov_size).float()
        rnn_init = (torch.zeros([1,1,150]), torch.zeros([1,1,150]))
        exp = 1, (image, 0, 0, image, 0, rnn_init )
        self.episode_memory.append(exp)

    def movement(self, action, location):
        """
        Takes an action and returns a new location
        """
        new_location = location
        if action == 0:
            new_location = (location[0] - 1, location[1], location[2])
        if action == 1:
            new_location = (location[0] + 1, location[1], location[2])
        if action == 2:
            new_location = (location[0], location[1] - 1, location[2])
        if action == 3:
            new_location = (location[0], location[1] + 1, location[2])
        return new_location

    def transition(self, env, models, action, location):
        """
        Changes the world based on the action taken
        """
        done = 0
        reward = 0
        new_loc = location
        attempted_locaton = self.movement(action, location)

        if action in [0, 1, 2, 3]:
            attempted_location_l0 = (attempted_locaton[0], attempted_locaton[1], 0)
            attempted_location_l1 = (attempted_locaton[0], attempted_locaton[1], 1)

            self.labour -= 2.1

            # below is repeated code because agents keep going on top of each other
            # and deleting each other.
            if isinstance(env.world[attempted_location_l0], Agent):
                reward = -0.1

            if isinstance(env.world[attempted_location_l1], Agent):
                reward = -0.1

            if isinstance(env.world[attempted_locaton], Agent):
                reward = -0.1

            if isinstance(env.world[attempted_location_l0], EmptyObject) and isinstance(
                env.world[attempted_location_l1], EmptyObject
            ):
                env.world[attempted_location_l1] = self
                new_loc = attempted_location_l1
                env.world[location] = EmptyObject()

            if isinstance(env.world[attempted_location_l0], Wall):
                reward = -0.1

            if isinstance(env.world[attempted_location_l0], House):
                reward = -0.1

            if isinstance(env.world[attempted_location_l0], Wood):
                if self.wood_skill < random.random():
                    # once this works, we need to set the reward to be 0 for collecting
                    # labour costs need to be implimented
                    reward = 1
                    self.wood += 1
                    env.world[attempted_location_l1] = self
                    env.world[attempted_location_l0] = EmptyObject()
                    env.world[location] = EmptyObject()
                    new_loc = attempted_location_l1

            if isinstance(env.world[attempted_location_l0], Stone):
                if self.stone_skill < random.random():
                    reward = 1
                    self.stone += 1
                    env.world[attempted_location_l1] = self
                    env.world[attempted_location_l0] = EmptyObject()
                    env.world[location] = EmptyObject()
                    new_loc = attempted_location_l1

        if action == 4:
            # note, you should not be able to build on top of another house
            reward = -0.1
            if self.stone > 0 and self.wood > 0 and self.house_skill < random.random():
                if isinstance(env.world[location[0], location[1], 0], EmptyObject):
                    reward = 20
                    self.stone -= 1
                    self.wood -= 1
                    env.world[location[0], location[1], 0] = House()

        if action == 5:  # sell wood
            if self.wood > 1:
                self.wood -= 1
                self.coin += 1
                reward = 5

        if action == 6:  # sell stone
            if self.stone > 1:
                self.stone -= 1
                self.coin += 1
                reward = 5

        if action == 7:  # buy wood
            if self.coin > 2:
                self.coin -= 2
                self.wood += 1
                reward = -1

        if action == 8:  # buy stone
            if self.coin > 2:
                self.coin -= 2
                self.stone += 1
                reward = -1

        if action == 9:  # do nothing
            pass

        next_state = env.pov(
            env.world,
            new_loc,
            self,
            inventory=[self.stone, self.wood, self.coin],
            layers=[0, 1],
        )
        self.reward += reward

        return env.world, reward, next_state, done, new_loc


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.health = 0  # wall stuff is basically empty
        self.appearance = [153.0, 51.0, 102.0]  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"
