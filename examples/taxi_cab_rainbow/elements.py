import random
from collections import deque
import numpy as np
import torch
import random
from gem.models.perception_singlePixel import agent_visualfield


class Wall:

    kind = "wall"  # class variable shared by all instances

    def __init__(self):
        self.appearance = [50.0, 50.0, 50.0]  # walls are purple
        self.passable = 0  # you can't walk through a wall
        self.action_type = "static"


class EmptyObject:

    kind = "empty"  # class variable shared by all instances

    def __init__(self):
        self.appearance = [0.0, 0.0, 0.0]  # empty is well, blank
        self.passable = 1  # whether the object blocks movement
        self.action_type = "empty"
        # self.change_appearance(0.05)

    def transition(self, world, location):
        generate_value = np.random.choice([0, 1], p=[0.9999, 0.0001])
        if generate_value == 1:
            world[location] = Passenger()
        return world

    def change_appearance(self, scaling):
        self.appearance = [
            random.random() * scaling,
            random.random() * scaling,
            random.random() * scaling,
        ]


class Passenger:

    kind = "passenger"  # class variable shared by all instances

    def __init__(self, world):
        super().__init__()
        self.appearance = (255.0, 0.0, 0.0)  # passengers are red
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"
        self.select_desired_location(world)
        # self.change_appearance(1)

    def change_appearance(self, scaling, max_value=255.0):
        R = min((random.random() * scaling + 235.0), max_value)
        G = min((random.random() * scaling + 0.0), max_value)
        B = min((random.random() * scaling + 0.0), max_value)
        self.appearance = [R, G, B]

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
        self.appearance = (0.0, 255.0, 0.0)  # destination is green
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"


class TaxiCab:

    kind = "taxi_cab"  # class variable shared by all instances

    def __init__(self, model):
        super().__init__()
        self.appearance = (255.0, 225.0, 0.0)  # taxi is yellow
        self.passable = 0  # whether the object blocks movement
        self.action_type = "neural_network"
        self.episode_memory = deque([], maxlen=10)  # we should read in these maxlens
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.has_passenger = 0
        self.driving_location = (0, 0, 0)
        self.init_rnn_state = None

    def init_replay(self, numberMemories, pov_size=9, visual_depth=4):
        """
        Fills in blank images for the LSTM before game play.
        """
        # pov_size = (self.vision * 2) - 1
        pov_size = 9
        visual_depth = 4  # change this to be 6 when we add the second layer of the task
        rnn_init = (torch.zeros([1, 1, 75]), torch.zeros([1, 1, 75]))
        image = torch.zeros(1, numberMemories, visual_depth, pov_size, pov_size).float()
        # exp = 1, (image, 0, 0, image, 0, rnn_init)
        exp = 1, (image, 0, 0, image, 0)
        self.episode_memory.append(exp)

    def pov(self, env, location, inventory=[], layers=[0]):
        """
        TODO: refactor all the code so that this is here
        """

        previous_state = self.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)
            img = agent_visualfield(env.world, loc, self.vision)
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)

        if len(inventory) > 0:
            """
            Loops through each additional piece of information and places into one layer
            """
            inventory_var = torch.tensor([])
            for item in range(len(inventory)):
                tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
                inventory_var = torch.cat((inventory_var, tmp), dim=0)
            inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
            state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state

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
        reward = -0.1
        new_loc = location
        attempted_locaton = self.movement(action, location)

        if action in [0, 1, 2, 3]:

            if isinstance(env.world[attempted_locaton], TaxiCab):
                reward = -2

            if isinstance(env.world[attempted_locaton], Wall):
                reward = -2

            if isinstance(env.world[attempted_locaton], EmptyObject):
                env.world[attempted_locaton] = self
                new_loc = attempted_locaton
                env.world[location] = EmptyObject()

            if isinstance(env.world[attempted_locaton], Passenger):
                self.driving_location = env.world[attempted_locaton].desired_location
                env.world[attempted_locaton] = self
                new_loc = attempted_locaton
                env.world[location] = EmptyObject()
                self.has_passenger = 255
                env.world[self.driving_location] = Destination()
                reward = -1

            if isinstance(env.world[attempted_locaton], Destination):
                reward = 25
                env.world[attempted_locaton] = self
                env.world[location] = EmptyObject()
                new_loc = attempted_locaton
                self.has_passenger = 0
                env.spawn_passenger()

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
        next_state = env.pov(new_loc, inventory=[self.has_passenger], layers=[0])

        self.reward += reward

        return env.world, reward, next_state, done, new_loc
