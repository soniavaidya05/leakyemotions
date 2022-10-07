from gem.environment.elements.element import Element
import random
from collections import deque
import numpy as np
import torch
from gem.environment.elements.element import Wall
from gem.environment.elements.element import EmptyObject


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.appearence = [153.0, 51.0, 102.0]  # walls are purple
        self.passable = 0  # you can't walk through a wall
        self.action_type = "static"


class EmptyObject:

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.appearence = [0.0, 0.0, 0.0]  # empty is well, blank
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"

    def transition(self, world, location):
        generate_value = np.random.choice([0, 1], p=[0.9999, 0.0001])
        if generate_value == 1:
            world[location] = Passenger()
        return world


class Passenger:

    kind = "passenger"  # class variable shared by all instances

    def __init__(self, world):
        super().__init__()
        self.appearence = (255, 0, 0)  # passengers are red
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"
        self.select_desired_location(world)

    def select_desired_location(self, world):
        """
        Returns the location of the passenger's desired destination
        """
        valid = False
        counter = 0
        while not valid:
            counter += 1
            x = random.randint(1, world.shape[0] - 2)
            y = random.randint(1, world.shape[1] - 2)
            z = 0
            location = (x, y, z)
            if isinstance(world[location], EmptyObject):
                valid = True
                self.desired_location = location
            if counter > 10:
                print("Error: could not find a valid location for passenger")
                break


class Destination:

    kind = "destination"  # class variable shared by all instances

    def __init__(self):
        super().__init__()
        self.appearence = (0, 255, 0)  # destination is green
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"


class TaxiCab:

    kind = "taxi_cab"  # class variable shared by all instances

    def __init__(self, model):
        super().__init__()
        self.appearence = (255, 225, 0)  # taxi is yellow
        self.passable = 0  # whether the object blocks movement
        self.action_type = "neural_network"
        self.episode_memory = deque([], maxlen=10)  # we should read in these maxlens
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.has_passenger = 0
        self.driving_location = (0, 0, 0)

    def init_replay(self, numberMemories):
        """
        Fills in blank images for the LSTM before game play.
        """
        # pov_size = (self.vision * 2) - 1
        pov_size = 9
        visual_depth = 4  # change this to be 6 when we add the second layer of the task
        image = torch.zeros(1, numberMemories, visual_depth, pov_size, pov_size).float()
        exp = 1, (image, 0, 0, image, 0)
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

    def transition(self, world, models, action, location):
        """
        Changes the world based on the action taken
        """
        done = 0
        reward = -1
        new_loc = location
        attempted_locaton = self.movement(action, location)

        if action in [0, 1, 2, 3]:

            if isinstance(world[attempted_locaton], TaxiCab):
                reward = -2

            if isinstance(world[attempted_locaton], Wall):
                reward = -2

            if isinstance(world[attempted_locaton], EmptyObject):
                world[attempted_locaton] = self
                new_loc = attempted_locaton
                world[location] = EmptyObject()

            if isinstance(world[attempted_locaton], Passenger):
                self.driving_location = world[attempted_locaton].desired_location
                world[attempted_locaton] = self
                new_loc = attempted_locaton
                world[location] = EmptyObject()
                self.has_passenger = 1
                world[self.driving_location] = Destination()
                reward = -1

            if isinstance(world[attempted_locaton], Destination):
                reward = 25
                world[attempted_locaton] = self
                world[location] = EmptyObject()
                new_loc = attempted_locaton
                self.has_passenger = 0

                # found a problem. transition may need to have the
                # whole environment passed to it if we want an action
                # to trigger an environment change (like spawn passenger)

        # the section below is probably the one that people
        # will have the most confusion about, since it is
        # using two features of Gem that we haven't talked about yet.
        # nameely, the inventory which is additional things that
        # can be added to a CNN (here it is whether a person is in the car)
        # and layers, which is by default just zero, but since I liked the idea
        # of the high and low res version of the world, we are going to need
        # to call both of them.

        next_state = models[self.policy].pov(
            world,
            new_loc,
            self,
            inventory=[self.has_passenger],
            layers=[0],
        )
        self.reward += reward

        return world, reward, next_state, done, new_loc
