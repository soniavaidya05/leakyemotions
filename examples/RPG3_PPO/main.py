# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)


from examples.RPG3_PPO.PPO_CPC import PPO, RolloutBuffer
from examples.RPG3_PPO.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video


from examples.RPG3_PPO.elements import EmptyObject, Wall

import numpy as np
import random
import torch

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")
SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """
    models = []
    models.append(
        PPO(
            device=device,
            state_dim=910,
            action_dim=4,
            lr_actor=0.001,  # .001
            lr_critic=0.005,  # .0005
            gamma=0.95,  # was .9
            K_epochs=10,  # was 10
            eps_clip=0.2,
        )
    )  # agent model 1

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models


world_size = 75

trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1

models = create_models()
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(),
    gem1p=0.03,
    gem2p=0.02,
    wolf1p=0.03,  # rename gem3p
)
# env.game_test()


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
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.02,
            gem3p=0.03,
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].episode_memory_PPO = RolloutBuffer()
            env.world[loc].init_replay(2)
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
            #        models[mods].qnetwork_target.load_state_dict(
            #            models[mods].qnetwork_local.state_dict()
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
                    state = env.pov(loc)
                    # params = (state.to(device), epsilon, env.world[loc].init_rnn_state)
                    # set up the right params below

                    action, action_logprob, init_rnn_state = models[
                        env.world[loc].policy
                    ].take_action(state, env.world[loc].init_rnn_state)
                    env.world[loc].init_rnn_state = init_rnn_state

                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                    ) = holdObject.transition(env, models, action[0], loc)

                    if reward == 15:
                        gems[0] = gems[0] + 1
                    if reward == 5:
                        gems[1] = gems[1] + 1
                    if reward == -5:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    reward = reward / 10

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

                    env.world[new_loc].episode_memory_PPO.states.append(state)
                    env.world[new_loc].episode_memory_PPO.actions.append(action)
                    env.world[new_loc].episode_memory_PPO.logprobs.append(
                        action_logprob
                    )

                    env.world[new_loc].episode_memory_PPO.rewards.append(reward)
                    env.world[new_loc].episode_memory_PPO.is_terminals.append(done)
                    env.world[new_loc].episode_memory.append(exp)

                    # env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "agent":
                        game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
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
                # models = transfer_world_memories(
                #    models, env.world, find_instance(env.world, "neural_network")
                # )

            if epoch > 200 and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    pass

                    # loss = models[env.world[loc].policy].training(
                    #    env.world[loc].episode_memory_PPO, entropy_coefficient=0.005
                    # )  # entropy_coefficient=0.01 was before
                    # env.world[loc].episode_memory_PPO.clear()
                    # losses = losses + loss.detach().cpu().numpy()

                    # experiences = models[mods].memory.sample()
                    # loss = models[mods].learn(experiences)
                    # losses = losses + loss
        if epoch > 100:
            agentList = find_instance(env.world, "neural_network")
            for loc in agentList:
                """
                Train the neural networks at the end of eac epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                loss = models[env.world[loc].policy].training(
                    env.world[loc].episode_memory_PPO, entropy_coefficient=0.005
                )  # entropy_coefficient=0.01 was before
                env.world[loc].episode_memory_PPO.clear()
                losses = losses + loss.detach().cpu().numpy()

                # experiences = models[mods].memory.sample()
                # loss = models[mods].learn(experiences)
                # losses = losses + loss

        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            # epsilon = update_epsilon(epsilon, turn, epoch)
            epsilon = max(epsilon - 0.00003, 0.2)

        if epoch % 100 == 0 and len(trainable_models) > 0 and epoch != 0:
            # print the state and update the counters. This should be made to be tensorboard instead
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
        gem1p=0.03,
        gem2p=0.02,
        gem3p=0.03,
    )
    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(1)
        env.world[loc].init_rnn_state = None

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
            """
            Reset the rewards for the trial to be zero for all agents
            """
            env.world[loc].reward = 0

        for loc in agentList:
            if env.world[loc].kind != "deadAgent":
                holdObject = env.world[loc]
                device = models[holdObject.policy].device
                state = env.pov(loc)
                params = (state.to(device), epsilon, env.world[loc].init_rnn_state)

                # set up the right params below

                action = models[env.world[loc].policy].take_action(state, 0)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                ) = holdObject.transition(env, models, action[0], loc)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


run_params = (
    [0.5, 2000, 20],
    [0.1, 10000, 20],
    [0.0, 10000, 20],
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
