import argparse
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    update_memories_rnn,
    find_agents,
    find_instance,
)
from examples.taxi_cab_rainbow.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)
from gem.models.iqn import IQNModel, PrioritizedReplay

from examples.taxi_cab_rainbow.env import TaxiCabEnv
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video
import torch
from tensorboardX import SummaryWriter
import time
import numpy as np

import random

# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"
save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "/Users/ethan/gem_output/"
logger = SummaryWriter(f"{save_dir}/taxicab/", comment=str(time.time))


# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

# device = "cpu"
print(device)

SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# The configuration of the network
# One of: "iqn", "iqn+per", "noisy_iqn", "noisy_iqn+per", "dueling", "dueling+per",
#         "noisy_dueling", "noisy_dueling+per"
NETWORK_CONFIG = "noisy_iqn"


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        IQNModel(
            state_size=torch.tensor([4, 9, 9]),
            action_size=4,
            network=NETWORK_CONFIG,
            munchausen=False,  # Don't use Munchausen RL loss
            layer_size=100,
            n_hidden_layers=3,
            n_step=1,  # Multistep IQN
            BATCH_SIZE=32,
            BUFFER_SIZE=1024,
            LR=0.00025,
            TAU=1e-3,  # Soft update parameter
            GAMMA=0.99,  # Discout factor
            N=12,  # Number of quantiles
            worker=1,  # number of parallel environments
            device=device,
            seed=SEED,
        )
    )  # taxi model

    return models


world_size = 10

trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 5
epsilon = 0.99

turn = 1

models = create_models()
env = TaxiCabEnv(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject,
)

# env.game_test()


def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
    world_size=10,
):
    """
    This is the main loop of the game
    """
    losses = 0
    game_points = [0, 0]
    for epoch in range(epochs):

        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
        )

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            pov_size = (
                env.tile_size[0] * (env.world[loc].vision * 2 + 1),
                env.tile_size[1] * (env.world[loc].vision * 2 + 1),
            )
            env.world[loc].init_replay(
                numberMemories=1, pov_size=pov_size, visual_depth=4
            )
            env.world[loc].init_rnn_state = None

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            # if epoch % sync_freq == 0:
            #    # update the double DQN model ever sync_frew
            #    for mods in trainable_models:
            #       models[mods].model2.load_state_dict(
            #            models[mods].model1.state_dict()
            #        )

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0

            for loc in agentList:
                if env.world[loc].kind != "deadAgent":

                    holdObject = env.world[loc]
                    device = models[holdObject.policy].device
                    state = env.pov(
                        loc, inventory=[holdObject.has_passenger], layers=[0]
                    )
                    params = (state.to(device), epsilon, env.world[loc].init_rnn_state)

                    # set up the right params below

                    action = models[env.world[loc].policy].take_action(state, epsilon)

                    # env.world[loc].init_rnn_state = init_rnn_state
                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                    ) = holdObject.transition(env, models, action, loc)

                    # these can be included on one replay

                    exp = (
                        # models[env.world[new_loc].policy].max_priority,
                        1,
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            # env.world[new_loc].init_rnn_state[0],
                            # env.world[new_loc].init_rnn_state[1],
                        ),
                    )

                    env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "taxi_cab":
                        game_points[0] = game_points[0] + reward
                    if env.world[new_loc].kind == "taxi_cab" and reward > 2:
                        game_points[1] = game_points[1] + 1

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
                # or reward > 0
            ):
                done = 1

            if len(trainable_models) > 0:
                """
                Update the next state and rewards for the agents after all have moved
                And then transfer the local memory to the model memory
                """
                # this updates the last memory to be the final state of the game board
                env.world = update_memories(
                    env,
                    find_instance(env.world, "neural_network"),
                    done,
                    end_update=False,  # the end update fails with non standard inputs. this needs to be fixed
                )

                # transfer the events for each agent into the appropriate model after all have moved
                models = transfer_world_memories(
                    models, env.world, find_instance(env.world, "neural_network")
                )

            if epoch > 200 and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    experiences = models[mods].memory.sample()
                    # print("experiences", len(experiences))
                    # print(experiences[0].shape)
                    loss = models[mods].learn(experiences)
                    losses = losses + loss

        for mods in trainable_models:
            """
            Train the neural networks at the end of eac epoch
            reduced to 64 so that the new memories ~200 are slowly added with the priority ones
            """
            # sample first
            # call learn fn on IQN: states, actions, rewards, next_states, dones = experiences

            experiences = models[mods].memory.sample()
            # print("experiences", len(experiences))
            # print(experiences[0].shape)
            loss = models[mods].learn(experiences)
            losses = losses + loss

        updateEps = True
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - (0.1 / epochs), 0.2)

        if epoch % 100 == 0 and len(trainable_models) > 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),
                round(game_points[1]),
                losses,
                epsilon,
                world_size,
            )
            # Tensorboard logging
            # logger.add_scalar('epoch', value=epoch, iteration=epoch)
            logger.add_scalar("num_turns", withinturn, epoch)
            logger.add_scalar("total_points", game_points[0], epoch)
            logger.add_scalar("n_passengers_delivered", game_points[1], epoch)
            logger.add_scalar("sum_loss", losses, epoch)
            logger.add_scalar("epsilon", epsilon, epoch)
            logger.add_scalar("world_size", world_size, epoch)

            game_points = [0, 0]
            losses = 0
    return models, env, turn, epsilon


# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


models = create_models()

run_params = (
    [0.99, 10, 100, 8],
    [0.9, 10000, 100, 8],
    [0.8, 10000, 100, 8],
    [0.7, 10000, 100, 8],
    [0.6, 10000, 100, 8],
    [0.5, 20000, 100, 8],
    [0.2, 20000, 100, 8],
)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        epsilon=run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
        world_size=run_params[modRun][3],
    )
    # save_models(
    #    models,
    #    save_dir,
    #    "taxi_cab_rainbow_" + str(modRun),
    # )
    # make_video(
    #    "taxi_cab_rainbow_" + str(modRun),
    #    save_dir,
    #    models,
    #    run_params[modRun][3],
    #    env,
    #    end_update=False,
    # )
