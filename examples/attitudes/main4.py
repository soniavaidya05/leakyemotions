# from tkinter.tix import Tree
from examples.attitudes.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)

from examples.attitudes.iRainbow_clean import iRainbowModel
from examples.attitudes.env_noStoredValue import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video


from examples.attitudes.elements import EmptyObject, Wall

import time
import numpy as np
import random
import torch

from sklearn.neighbors import NearestNeighbors

# save_dir = "/Users/yumozi/Projects/gem_data/RPG3_test/"
save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")
SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

from collections import deque, namedtuple

# noisy_dueling

from scipy.spatial import distance
import numpy as np
import random


# If True, use the KNN model when computing k-most similar recent states. Otherwise, use a brute-force search.
USE_KNN_MODEL = True
# Run profiling on the RL agent to see how long it takes per step
RUN_PROFILING = False

print(f"Using device: {device}")
print(f"Using KNN model: {USE_KNN_MODEL}")
print(f"Running profiling: {RUN_PROFILING}")


def k_most_similar_recent_states(
    state, knn: NearestNeighbors, memories, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        distances = [distance.euclidean(state, memory[0]) for memory in memories]
        k_indices = np.argsort(distances)[:k]

    # Create a list of weights based on the decay rate, but only for the k most similar states
    weights = [decay_rate**i for i in k_indices]

    # Normalize the weights
    weights = [w / sum(weights) for w in weights]

    # Sample from the k most similar memories with the given weights
    sampled_memories = random.choices(
        [memories[i] for i in k_indices], weights=weights, k=k
    )

    return sampled_memories


from sklearn.neighbors import NearestNeighbors


def average_reward(memories):
    # Extract the rewards from the tuples (assuming reward is the second element in each tuple)
    rewards = [memory[1] for memory in memories]

    # Calculate the average reward
    average_reward = sum(rewards) / len(rewards)

    return average_reward


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        iRainbowModel(
            in_channels=8,
            num_filters=8,
            cnn_out_size=567,  # 910
            state_size=torch.tensor(
                [8, 9, 9]
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


world_size = 25

trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1

object_memory = deque(maxlen=5000)
state_knn = NearestNeighbors(n_neighbors=5)

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
    change=False,
    masked_attitude=False,
    attitude_layer=True,
):
    """
    This is the main loop of the game
    """
    losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]
    decay_rate = 0.5  # Adjust as needed

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
            change=change,
            masked_attitude=masked_attitude,
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(1)
            env.world[loc].init_rnn_state = None

        turn = 0

        start_time = time.time()

        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].qnetwork_target.load_state_dict(
                        models[mods].qnetwork_local.state_dict()
                    )

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
                    batch, timesteps, channels, height, width = state.shape

                    if attitude_layer:
                        if len(object_memory) > 50:
                            for t in range(timesteps):
                                for h in range(height):
                                    for w in range(width):
                                        state[0, t, 7, h, w] = 0
                                        # if env.world[h, w, 0].kind != "empty":
                                        object_state = state[0, t, :7, h, w]
                                        mems = k_most_similar_recent_states(
                                            object_state,
                                            state_knn,
                                            object_memory,
                                            1,
                                            k=5,
                                        )

                                        r = average_reward(mems)
                                        state[0, t, 7, h, w] = r * 255
                                        # if random.random() < 0.1:
                                        #    print(mems)
                                        #    print(r, state[0, t, :, h, w])

                        # if random.random() < 0.01:
                        #    for t in range(timesteps):
                        #        for h in range(height):
                        #            for w in range(width):
                        #                print(t, h, w, state[0, t, :, h, w])

                    # channels = state[0, 0, :7, 0, 0]
                    # result_tuple = tuple(map(float, channels))

                    # params = (state.to(device), epsilon, env.world[loc].init_rnn_state)
                    # set up the right params below

                    action = models[env.world[loc].policy].take_action(state, epsilon)

                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        object_info,
                    ) = holdObject.transition(env, models, action[0], loc)

                    # create object memory
                    state_object = object_info[0:7]
                    if attitude_layer:
                        object_exp = (state_object, reward)
                    else:
                        object_exp = (state_object, 0)
                    object_memory.append(object_exp)
                    if USE_KNN_MODEL:
                        # Fit a k-NN model to states extracted from the replay buffer
                        state_knn.fit([exp[0] for exp in object_memory])

                    if reward == 15:
                        gems[0] = gems[0] + 1
                    if reward == 5:
                        gems[1] = gems[1] + 1
                    if reward == -5:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    # these can be included on one replay

                    if attitude_layer:
                        if len(object_memory) > 50:
                            for t in range(timesteps):
                                for h in range(height):
                                    for w in range(width):
                                        next_state[0, t, 7, h, w] = 0
                                        # if (
                                        #    env.world[h, w, 0].kind != "empty"
                                        #    or env.world[h, w, 0].kind != "agent"
                                        # ):
                                        object_state = state[0, t, :7, h, w]
                                        mems = k_most_similar_recent_states(
                                            object_state,
                                            state_knn,
                                            object_memory,
                                            1,
                                            k=5,
                                        )
                                        r = average_reward(mems)
                                        next_state[0, t, 7, h, w] = r * 255
                                        r = 0

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

        end_time = time.time()
        if RUN_PROFILING:
            print(f"Epoch {epoch} took {end_time - start_time} seconds")

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


# import matplotlib.animation as animation
# from gem.models.perception import agent_visualfield
#


def load_attitudes(
    state, env=env, object_memory=object_memory, state_knn=state_knn, decay_rate=0.5
):
    batch, timesteps, channels, height, width = state.shape
    if len(object_memory) > 50:
        for t in range(timesteps):
            for h in range(height):
                for w in range(width):
                    state[0, t, 7, h, w] = 0
                    object_state = state[0, t, :7, h, w]
                    mems = k_most_similar_recent_states(
                        object_state,
                        state_knn,
                        object_memory,
                        decay_rate,
                        k=5,
                    )
                    r = average_reward(mems)
                    state[0, t, 7, h, w] = r * 255
    return state


def test_memory(env=env, object_memory=object_memory, new_env=False):
    if new_env:
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.02,
            gem3p=0.03,
        )
    agentList = find_instance(env.world, "neural_network")
    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(1)
        env.world[loc].init_rnn_state = None

    for loc in find_instance(env.world, "neural_network"):
        state = env.pov(loc)

        batch, timesteps, channels, height, width = state.shape
        state_attitudes = load_attitudes(state)
        for i in range(state.shape[3]):
            for j in range(state.shape[4]):
                for t in range(state.shape[1]):
                    i2 = loc[0] - 4 + i
                    j2 = loc[1] - 4 + j
                    flagged = False
                    if i2 < 0:
                        flagged = True
                    if i2 >= world_size:
                        flagged = True
                    if j2 < 0:
                        flagged = True
                    if j2 >= world_size:
                        flagged = True

                    if flagged:
                        object_kind = "outside"
                        object_value = 0
                    else:
                        object_kind = env.world[i2, j2, 0].kind
                        object_value = env.world[i2, j2, 0].value

                    print(loc, t, i, j, object_kind, state[0, t, 7, i, j], object_value)


models = create_models()
# for mod in range(len(models)):
#    models[mod].new_memory_buffer(1024, SEED, 3)

run_params = (
    # [0.5, 50, 20, False, False, True],
    [0.5, 500, 20, False, False, True],
    [0.1, 500, 20, False, False, True],
    [0.0, 200, 20, False, False, True],
    [0.0, 1000, 20, True, False, True],
    [0.0, 1000, 20, True, False, True],
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
        change=run_params[modRun][3],
        masked_attitude=run_params[modRun][4],
        attitude_layer=run_params[modRun][5],
    )
    test_memory(env, object_memory, new_env=False)
    # for mod in range(len(models)):
    #    models[mod].new_memory_buffer(1024, SEED, 3)
