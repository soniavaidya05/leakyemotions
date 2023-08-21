from abc import ABC
from collections import deque
import numpy as np
import torch


class Element(ABC):
    def __init__(self):
        self.appearance = None  # how to display this agent
        self.vision = None  # visual radius
        self.policy = None  # policy function or model
        self.value = None  # reward given to another agent
        self.reward = None  # reward received on this trial
        self.static = True  # whether the object gets to take actions or not
        self.passable = False  # whether the object blocks movement
        self.trainable = False  # whether there is a network to be optimized
        self.has_transitions = False


class EmptyObject:
    kind = "empty"  # class variable shared by all instances

    def __init__(self, app_size):
        self.app_size = app_size
        self.health = 0  # empty stuff is basically empty
        self.appearance = np.zeros(self.app_size)  # empty is well, blank
        self.vision = 1  # empty stuff is basically empty
        self.policy = "NA"  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.reward = 0  # empty stuff is basically empty
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"


class Wall:
    kind = "wall"  # class variable shared by all instances

    def __init__(self, app_size):
        self.app_size = app_size
        self.health = 0  # wall stuff is basically empty
        self.appearance = np.zeros(self.app_size)  # walls are purple
        self.vision = 0  # wall stuff is basically empty
        self.policy = "NA"  # walls do not do anything
        self.value = 0  # wall stuff is basically empty
        self.reward = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0  # whether the object is deterministic
        self.action_type = "static"  # rays disappear after one turn
        self.appearance[1] = 255.0


class Gem:
    kind = "gem"  # class variable shared by all instances

    def __init__(self, value, color, app_size):
        super().__init__()
        self.health = 1  # for the gem, whether it has been mined or not
        self.appearance = color  # gems are green
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.reward = 0  # how much reward this gem has found (will remain 0)
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.trainable = 0  # whether there is a network to be optimized
        self.has_transitions = False
        self.deterministic = 0
        self.action_type = "static"
        self.wood = value[0]
        self.stone = value[1]

        # game variables
        self.good_person = True
        self.approached = 0


class Agent:
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model, app_size):
        self.app_size = app_size
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = np.zeros(self.app_size)  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"
        self.appearance[0] = 255.0
        self.stone = 0
        self.wood = 0

    def init_replay(self, numberMemories, pov_size=9, visual_depth=3):
        """
        Fills in blank images for the LSTM before game play.
        Implicitly defines the number of sequences that the LSTM will be trained on.
        """
        # pov_size = 9 # this should not be needed if in the input above
        image = torch.zeros(
            1, numberMemories, self.app_size, pov_size, pov_size
        ).float()
        priority = torch.tensor(0.1)
        blank = torch.tensor(0.0)
        exp = (priority, (image, blank, blank, image, blank))
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
        # Set up variables
        done = 0
        reward = 0

        # Get the agent
        agent = env.world[location]

        new_loc = location
        attempted_location = self.movement(action, location)
        new_object = env.world[attempted_location]

        # Default resource outcome is "nothing"
        resource_outcome = [1, 0, 0]

        # If the agent can move through the new location...
        if new_object.passable == 1:

            # Clear the agent's previous location
            env.world[location] = EmptyObject(env.app_size)

            # If the object is a gem...
            if new_object.kind == "gem":
            
                # For contextual environments, agents are rewarded based on their current state
                # They alternately want wood or stone
                if env.contextual:

                    # Check self.wood and self.stone
                    if self.wood > 0:
                        if new_object.stone > 0:
                            # If the new object has stone, build house (reward) and set resources to 0
                            resource_outcome = [0, 0, 1]
                            reward = 10
                            self.stone, self.wood = 0, 0
                        elif new_object.wood > 0:
                            # Otherwise, you just get more wood
                            resource_outcome = [0, 1, 0]
                            reward = 0
                            self.wood = 1 # Owning wood is a binary here, you either have it or you don't
                    elif self.stone > 0:
                        if new_object.stone > 0:
                            # If the new object has stone, nothing happens
                            resource_outcome = [0, 0, 1]
                            reward = 0
                            self.stone = 1
                        elif new_object.wood > 0:
                            # If it has stone, then you get rewards and resources reset
                            resource_outcome = [0, 1, 0]
                            reward = 10
                            self.stone, self.wood = 0, 0
                    else:
                        # If the agent has nothing, just add stone or wood
                        if new_object.stone > 0:
                            resource_outcome = [0, 0, 1]
                            self.stone = 1
                        elif new_object.wood > 0:
                            resource_outcome = [0, 1, 0]
                            self.wood = 1
                
                # For noncontextual environments, agents always want wood
                else:

                    # 10 reward if wood, 0 otherwise
                    if new_object.wood > 0:
                        resource_outcome = [0, 1, 0]
                        reward = 10
                    elif new_object.stone > 0:
                        resource_outcome = [0, 0, 1]
                        reward = 0

            # Finally, move the agent to the new location
            env.world[attempted_location] = agent
            new_loc = attempted_location

        # If the location is impassable, check whether it's a wall
        # Punish if the agent has run into a wall
        else:
            if isinstance(
                env.world[attempted_location], Wall
            ):
                reward = -1

        next_state = env.pov(new_loc)
        self.reward += reward

        # if reward == -1:
        #    print("You hit a wall!", object_info)

        return (
            env.world,
            reward,
            next_state,
            done,
            new_loc,
            new_object.appearance,
            resource_outcome,
        )
