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
        """
        Fills in blank images for the LSTM before game play.
        Impicitly defines the number of sequences that the LSTM will be trained on.
        """
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

            # BELOW MAY BE EXTRA ACCIDENTAL CODE
            # if world[attempted_locaton_1, attempted_locaton_2, 0].appearence == [
            #    0.0,
            #    0.0,
            #    255.0,
            # ]:
            #    reward = 10
            #    wolfEats = wolfEats + 1

            world[i, j, 0] = EmptyObject()
            world[attempted_locaton_1, attempted_locaton_2, 0] = self
            new_locaton_1 = attempted_locaton_1
            new_locaton_2 = attempted_locaton_2
            reward = 0
        else:
            if isinstance(world[attempted_locaton_1, attempted_locaton_2, 0], Wall):
                reward = -0.1
            if isinstance(world[attempted_locaton_1, attempted_locaton_2, 0], Agent):
                """
                If the wolf and the agent are in the same location, the agent dies.
                In addition to giving the wolf a reward, the agent also gets a punishment.
                TODO: This needs to be updated to be in the Agent class rather than here
                """
                reward = 10
                game_points[1] = game_points[1] + 1
                newVersion = 0
                if newVersion == 0:

                    # update the last memory of the agent that was eaten

                    lastexp = world[attempted_locaton_1, attempted_locaton_2, 0].replay[
                        -1
                    ]
                    world[attempted_locaton_1, attempted_locaton_2, 0].replay[-1] = (
                        lastexp[0],
                        lastexp[1],
                        -25,
                        lastexp[3],
                        1,
                    )

                    # TODO: Below is very clunky and a more principles solution needs to be found

                    if ModelType == "DQN":
                        models[
                            world[attempted_locaton_1, attempted_locaton_2, 0].policy
                        ].transfer_memories(
                            world,
                            attempted_locaton_1,
                            attempted_locaton_2,
                            extra_reward=True,
                        )
                    if ModelType == "AC":
                        # note, put in the whole code for updatng an AC model here
                        if (
                            len(
                                world[
                                    attempted_locaton_1, attempted_locaton_2, 0
                                ].AC_value
                            )
                            > 0
                        ):

                            finalReward = torch.tensor(-25).float().reshape(1, 1)

                            if (
                                world[
                                    attempted_locaton_1, attempted_locaton_2, 0
                                ].AC_reward.shape
                                == world[
                                    attempted_locaton_1, attempted_locaton_2, 0
                                ].AC_value.shape
                            ):
                                world[
                                    attempted_locaton_1, attempted_locaton_2, 0
                                ].AC_reward[-1] = finalReward
                            else:
                                world[
                                    attempted_locaton_1, attempted_locaton_2, 0
                                ].AC_reward = torch.concat(
                                    [
                                        world[
                                            attempted_locaton_1, attempted_locaton_2, 0
                                        ].AC_reward,
                                        finalReward,
                                    ]
                                )
                                models[
                                    world[
                                        attempted_locaton_1, attempted_locaton_2, 0
                                    ].policy
                                ].transfer_memories_AC(
                                    world, attempted_locaton_1, attempted_locaton_2
                                )

                    world[attempted_locaton_1, attempted_locaton_2, 0] = DeadAgent()
                if newVersion == 1:
                    world = world[attempted_locaton_1, attempted_locaton_2, 0].died(
                        models,
                        world,
                        attempted_locaton_1,
                        attempted_locaton_2,
                        extra_reward=True,
                    )
                    world[attempted_locaton_1, attempted_locaton_2, 0] = DeadAgent()

        if update_experience_buffer == True:
            """
            Put the transition into the experience buffer
            """
            input2 = models[self.policy].pov(world, new_locaton_1, new_locaton_2, self)
            exp = (input, action, reward, input2, done)
            self.replay.append(exp)
            self.reward += reward

        return world, models, game_points

    def transitionSingle(self, world, models, action, location):
        i, j, k = location
        done = 0

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
            world[attempted_locaton_1, attempted_locaton_2, 0] = self
            new_locaton_1 = attempted_locaton_1
            new_locaton_2 = attempted_locaton_2
            reward = 0
        else:
            if isinstance(world[attempted_locaton_1, attempted_locaton_2, 0], Wall):
                reward = -0.1
            if isinstance(world[attempted_locaton_1, attempted_locaton_2, 0], Agent):
                """
                If the wolf and the agent are in the same location, the agent dies.
                In addition to giving the wolf a reward, the agent also gets a punishment.
                TODO: This needs to be updated to be in the Agent class rather than here
                """
                reward = 10

                # TODO: the died function is not working properly

                exp = world[attempted_locaton_1, attempted_locaton_2, 0].replay[-1]
                exp = (exp[0], exp[1], -25, exp[3], 1)
                world[attempted_locaton_1, attempted_locaton_2, 0].replay[-1] = exp
                models[
                    world[attempted_locaton_1, attempted_locaton_2, 0].policy
                ].transfer_memories(
                    world, attempted_locaton_1, attempted_locaton_2, extra_reward=True
                )

                # world = world[attempted_locaton_1, attempted_locaton_2, 0].died(
                #        models,
                #        world,
                #        attempted_locaton_1,
                #        attempted_locaton_2,
                #        extra_reward=True,
                #    )
                world[attempted_locaton_1, attempted_locaton_2, 0] = DeadAgent()

        next_state = models[self.policy].pov(world, new_locaton_1, new_locaton_2, self)

        return world, reward, next_state, done, (new_locaton_1, new_locaton_2, 0)
