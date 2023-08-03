# Set module path
import os
import sys
module_path = os.path.abspath('../..')
if module_path not in sys.path:
    sys.path.append(module_path)
    print(sys.path)

# Import basic packages
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import argparse

# Import utils
from gem.utils import (
    update_memories,
    transfer_world_memories,
    find_instance,
)

from IPython.display import clear_output

# Import model and environment
from iRainbow_clean import iRainbowModel
from env import RPG
from elements import EmptyObject, Wall

# Set up tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Set seed for replication
SEED = 1 
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# choose device
device = 'cpu'
# if torch.backends.mps.is_available():
#    device = torch.device("mps")



def create_models(n_agents = 1):
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    for i in range(n_agents):
        models.append(
            iRainbowModel(
                in_channels=7,
                num_filters=7,
                cnn_out_size=567,  # 910
                state_size=torch.tensor(
                    [7, 9, 9]
                ),  # this seems to only be reading the first value
                action_size=4,
                layer_size=250,  # 100
                n_step=3,  # Multistep IQN (rainbow paper uses 3)
                BATCH_SIZE=64,
                BUFFER_SIZE=1024,
                LR=0.00025,  # 0.00025
                TAU=1e-3,  # Soft update parameter
                GAMMA=0.95,  # Discout factor 0.99
                N=12,  # Number of quantiles
                device=device,
                seed=SEED,
            )
        )

    return models


world_size = 11

trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1

# Set up model and environment
models = create_models(n_agents = len(trainable_models))
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    truck_prefs=(10,5,-5),
    baker_mode=True
)

def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
):
    """
    This is the main loop of the game
    """
    losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]
    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the reset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            agent = env.world[loc] # Agent at this location
            agent.init_replay(2)
            agent.init_rnn_state = None

        while done == 0:
            """
            While the agent is not done, move the agent
            """
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].qnetwork_target.load_state_dict(
                        models[mods].qnetwork_local.state_dict()
                    )

            # Find agents in the environment; they move in a randomized order
            agentList = find_instance(env.world, "neural_network")
            random.shuffle(agentList)

            # For each agent, act and get the transitions and experience
            for loc in agentList:
                agent = env.world[loc]
                agent.reward = 0
                state = env.pov(loc)

                action = models[agent.policy].take_action(state, epsilon)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                ) = agent.transition(env, models, action[0], loc)

                if reward == 10:
                    gems[0] = gems[0] + 1
                if reward == 5:
                    gems[1] = gems[1] + 1
                if reward == -5:
                    gems[2] = gems[2] + 1
                if reward == -1:
                    gems[3] = gems[3] + 1

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
                    ),
                )

                env.world[new_loc].episode_memory.append(exp)

                if env.world[new_loc].kind == "agent":
                    game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
                or reward in env.truck_prefs
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
                    end_update=True,
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
                    loss = models[mods].learn(experiences)
                    losses = losses + loss

        if epoch > 100:
            for mods in trainable_models:
                """
                Train the neural networks at the end of eac epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                experiences = models[mods].memory.sample()
                loss = models[mods].learn(experiences)
                losses = losses + loss

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - 0.00003, 0.2)

        if len(trainable_models) > 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            writer.add_scalar("num_turns", withinturn, epoch)
            writer.add_scalar("Losses", losses, epoch)
            writer.add_scalar("Game Points", game_points[0], epoch)
            writer.add_scalar("Korean Truck (+10)", gems[0], epoch)
            writer.add_scalar("Lebanese Truck (+5)", gems[1], epoch)
            writer.add_scalar("Mexican Truck (-5)", gems[2], epoch)
            writer.add_scalar("Wall Collisions", gems[3], epoch)

            if epoch % 50 == 0 and epoch != 0:

                clear_output(wait = True)
                
                agentList = find_instance(env.world, "neural_network")

                for loc in agentList:
                    for i in range(env.height):
                        print(f'{" ".join([str(j)[0] for j in env.world[i, :, 0]])}')

                print(
                    epoch,
                    withinturn,
                    round(game_points[0]),
                    gems,
                    losses,
                    epsilon,
                )

            game_points = [0, 0]
            gems = [0, 0, 0, 0]
            losses = 0

            

    return models, env, turn, epsilon


# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


models = create_models()


import matplotlib.animation as animation
from gem.models.perception import agent_visualfield


def eval_game(models, env, turn, epsilon, epochs=10000, max_turns=100, filename="tmp"):
    """
    This is the main loop of the game
    """
    game_points = [0, 0]

    fig = plt.figure()
    ims = []
    env.reset_env(world_size, world_size)

    """
    Move each agent once and then update the world
    Creates new gamepoints, resets agents, and runs one episode
    """

    done = 0

    # create a new gameboard for each epoch and repopulate
    # the resset does allow for different params, but when the world size changes, odd
    env.reset_env(
        height=world_size,
        width=world_size,
        layers=1,
        gem1p=0.0,
        gem2p=0.0,
        gem3p=0.0,
    )

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        agent = env.world[loc] # Agent at this location
        agent.init_replay(1)
        agent.init_rnn_state = None

    for _ in range(max_turns):
        """
        Find the agents and wolves and move them
        """

        image = agent_visualfield(env.world, (0, 0), env.tile_size, k=None)
        im = plt.imshow(image, animated=True)
        ims.append([im])

        agentList = find_instance(env.world, "neural_network")

        random.shuffle(agentList)

        for loc in agentList:
            agent = env.world[loc] # Agent at this location
            agent.reward = 0 # Reset the agent's reward
            device = models[agent.policy].device
            state = env.pov(loc)
            params = (state.to(device), epsilon, agent.init_rnn_state)

            # set up the right params below

            action = models[agent.policy].take_action(state, 0)

            (
                env.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = agent.transition(env, models, action[0], loc)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


run_params = (
    [0.5, 2000, 100],
    [0.1, 10000, 100],
    [0.0, 10000, 100],
)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
    )
    # filename = save_dir + "RPG2d_" + str(modRun) + ".gif"
    # eval_game(models, env, turn, 0, 1, 35, filename)

    # save_models(
    #    models,
    #    save_dir,
    #    "WolvesGems_" + str(modRun),
    # )
    # make_video(
    #    "WolvesGems_" + str(modRun),
    #    save_dir,
    #    models,
    #    20,
    #    env,
    # )
