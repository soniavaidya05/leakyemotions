from gem.environment.elements.element import Element
import random
from collections import deque
import numpy as np
import torch
from gem.environment.elements.element import Wall


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


class Agent:

    kind = "agent"  # class variable shared by all instances

    def __init__(self, model, appearence, wood_skill, stone_skill, house_skill):
        self.appearence = appearence  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.static = 0  # whether the object gets to take actions or not
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.replay = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.deterministic = 0  # whether the object is deterministic
        self.stone = 0
        self.wood = 0
        self.coin = 0
        self.wood_skill = wood_skill
        self.stone_skill = stone_skill
        self.house_skill = house_skill
        self.selling_wood = 0
        self.selling_stone = 0

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
        num_elements = 5  # coin + wood + stone + person1 + person2
        person_depth = 8  # how  many elements to define a person
        image = torch.zeros(1, numberMemories, person_depth, num_elements, 1).float()
        exp = 1, (image, 0, 0, image, 0)
        self.replay.append(exp)

    def transition(self, world, action, selected_person):
        """
        Changes the world based on the action taken
        """
        done = 0
        reward = 0

        if action == 1:
            if self.wood_skill > random.ramdint(0, 100):
                self.wood += 1
        if action == 2:
            if self.stone_skill > random.ramdint(0, 100):
                self.wood += 1
        if action == 3:
            if self.wood > 0 and self.stone > 0:
                if self.house_skill > random.ramdint(0, 100):
                    self.wood -= 1
                    self.stone -= 1
                    self.coin += 100
        if action == 4:
            self.selling_stone = 1
        if action == 5:
            self.selling_wood = 1
        if action == 6:
            if selected_person.selling_stone == 1:
                self.coin = -10
                self.stone += 1
                selected_person.coin += 10
                selected_person.stone -= 1
        if action == 7:
            if selected_person.selling_wood == 1:
                self.coin = -10
                self.wood += 1
                selected_person.coin += 10
                selected_person.wood -= 1

        return selected_person


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
