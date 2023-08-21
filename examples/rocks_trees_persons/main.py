# from tkinter.tix import Tree
from examples.rocks_trees_persons.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
    plot_time_decay,
)
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from examples.rocks_trees_persons.iRainbow_clean import iRainbowModel
from examples.rocks_trees_persons.env import RPG
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video

import torch.optim as optim
from examples.rocks_trees_persons.elements import EmptyObject, Wall

import time
import numpy as np
import random
import torch

from collections import deque, namedtuple
from scipy.spatial import distance

from datetime import datetime

from sklearn.neighbors import NearestNeighbors


# save_dir = "/Users/yumozi/Projects/gem_data/RPG3_test/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
save_dir = "/Users/ethan/attitudes-output"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

import time


def k_most_similar_recent_states(
    state, knn: NearestNeighbors, memories, object_memory_states_tensor, decay_rate, k=5
):
    if USE_KNN_MODEL:
        # Get the indices of the k most similar states (without selecting them yet)
        state = state.cpu().detach().numpy().reshape(1, -1)
        k_indices = knn.kneighbors(state, n_neighbors=k, return_distance=False)[0]
    else:
        # Perform a brute-force search for the k most similar states
        # distances = [distance.euclidean(state, memory[0]) for memory in memories]
        # k_indices = np.argsort(distances)[:k]

        # let's try another way using torch operations...

        # Calculate the squared Euclidean distance
        squared_diff = torch.sum((object_memory_states_tensor - state) ** 2, dim=1)
        # Take the square root to get the Euclidean distance
        distance = torch.sqrt(squared_diff)
        # Argsort and take top-k
        k_indices = torch.argsort(distance, dim=0)[:k]

    # Gather the k most similar memories based on the indices, preserving the order
    most_similar_memories = [memories[i] for i in k_indices]

    return most_similar_memories


def compute_weighted_average(
    state, memories, similarity_decay_rate=1, time_decay_rate=1
):
    if not memories:
        return 0

    memory_states, rewards = zip(*memories)
    memory_states = np.array(memory_states)
    state = np.array(state)

    # Compute Euclidean distances
    distances = np.linalg.norm(memory_states - state, axis=1)
    max_distance = np.max(distances) if distances.size else 1

    # Compute similarity weights with exponential decay
    similarity_weights = (
        np.exp(-distances / max_distance * similarity_decay_rate)
        if max_distance != 0
        else np.ones_like(distances)
    )

    # Compute time weights with exponential decay
    N = len(memories)
    time_weights = np.exp(-np.arange(N) / (N - 1) * time_decay_rate)

    # Combine the weights
    weights = similarity_weights * time_weights

    # Compute the weighted sum
    weighted_sum = np.dot(weights, rewards)
    total_weight = np.sum(weights)

    return weighted_sum / total_weight if total_weight != 0 else 0


SEED = time.time()  # Seed for replicating training runs
# np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# If True, use the KNN model when computing k-most similar recent states. Otherwise, use a brute-force search.
USE_KNN_MODEL = True
# Run profiling on the RL agent to see how long it takes per step
RUN_PROFILING = False

print(f"Using device: {device}")
print(f"Using KNN model: {USE_KNN_MODEL}")
print(f"Running profiling: {RUN_PROFILING}")


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class ResourceModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
    ):
        super(ResourceModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3)  # Three outputs for three classes

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        probabilities = torch.softmax(self.fc4(x), dim=-1)
        return probabilities

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32, class_weights=False):
        if class_weights:
            # Calculate class weights
            all_outcomes = [outcome for _, outcome in self.replay_buffer]
            num_samples = len(all_outcomes)
            class_counts = [sum([out[i] for out in all_outcomes]) for i in range(3)]

            # Adding a small epsilon to prevent division by zero
            epsilon = 1e-10
            class_weights = torch.tensor(
                [(num_samples / (count + epsilon)) for count in class_counts]
            ).to(torch.float32)

            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Weighted Cross-Entropy Loss
                loss = F.cross_entropy(
                    probabilities, torch.argmax(targets, dim=1), weight=class_weights
                )

                loss.backward()
                self.optimizer.step()

        else:
            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Cross-Entropy Loss without weights
                loss = F.cross_entropy(probabilities, torch.argmax(targets, dim=1))

                loss.backward()
                self.optimizer.step()

        return loss.item()

    def add_memory(self, state, outcome):
        self.replay_buffer.append((state, outcome))


class ValueModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
        num_tau=32,
    ):
        super(ValueModel, self).__init__()
        self.num_tau = num_tau
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_tau)

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        taus = torch.linspace(0, 1, steps=self.num_tau, device=x.device).view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        quantiles = self.fc4(x)

        # Extract the 25th, 50th, and 75th percentiles
        percentiles = quantiles[
            :,
            [
                int(self.num_tau * 0.1) - 1,
                int(self.num_tau * 0.5) - 1,
                int(self.num_tau * 0.9) - 1,
            ],
        ]
        return percentiles, taus

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32):
        for _ in range(len(memories) // batch_size):
            batch = random.sample(memories, batch_size)
            states, rewards = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            rewards = (
                torch.tensor(rewards, dtype=torch.float32)
                .view(-1, 1)
                .repeat(1, self.num_tau)
            )

            self.optimizer.zero_grad()
            # Forward pass to get all quantiles, not just the 25th, 50th, and 75th percentiles
            x = torch.relu(self.fc1(states))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            quantiles = self.fc4(x)  # Shape [batch_size, num_tau]

            errors = rewards - quantiles
            huber_loss = torch.where(
                errors.abs() < 1, 0.5 * errors**2, errors.abs() - 0.5
            )
            taus = (
                torch.linspace(0, 1, steps=self.num_tau, device=states.device)
                .view(1, -1)
                .repeat(batch_size, 1)
            )
            quantile_loss = (taus - (errors < 0).float()).abs() * huber_loss
            loss = quantile_loss.mean()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def add_memory(self, state, reward):
        self.replay_buffer.append((state, reward))


value_model = ValueModel(state_dim=10, memory_size=250)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        iRainbowModel(
            in_channels=5,
            num_filters=5,
            cnn_out_size=567,  # 910
            state_size=torch.tensor(
                [5, 9, 9]
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
value_losses = []
trainable_models = [0]
sync_freq = 200  # https://openreview.net/pdf?id=3UK39iaaVpE
modelUpdate_freq = 4  # https://openreview.net/pdf?id=3UK39iaaVpE
epsilon = 0.99

turn = 1


def eval_attiude_model(value_model=value_model):
    atts = []
    s = torch.zeros(7)
    r = value_model(s)
    atts.append(round(r.item(), 2))
    for a in range(7):
        s = torch.zeros(7)
        s[a] = 255.0
        r = value_model(s)
        atts.append(round(r.item(), 2))
    return atts


object_exp2 = deque(maxlen=2500)
object_memory = deque(maxlen=250)
state_knn = NearestNeighbors(n_neighbors=100)
state_knn_CMS = NearestNeighbors(n_neighbors=100)

models = create_models()
env = RPG(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject(11),
    gem1p=0.03,
    gem2p=0.03,
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
    epsilon_decay=0.9999,
    attitude_condition="implicit_attitude",
    switch_epoch=1000,
    episodic_decay_rate=1.0,
    similarity_decay_rate=1.0,
):
    """
    This is the main loop of the game
    """
    losses = 0
    local_value_losses = 0
    game_points = [0, 0]
    gems = [0, 0, 0, 0]
    decay_rate = 0.2  # Adjust as needed
    change = True
    gem_changes = 0

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        if epoch % switch_epoch == 0:
            gem_changes = gem_changes + 1
            env.change_gem_values()
        epsilon = epsilon * epsilon_decay
        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=1,
            gem1p=0.03,
            gem2p=0.03,
            gem3p=0.03,
            change=change,
        )

        working_memory = 1

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(working_memory)
            env.world[loc].init_rnn_state = None

        agent_wood = env.world[loc].wood
        agent_stone = env.world[loc].stone

        # reset the variables to be safe

        for i in range(world_size):
            for j in range(world_size):
                env.world[i, j, 0].appearance[-3] = 0.0
                env.world[i, j, 0].appearance[-2] = 0.0
                env.world[i, j, 0].appearance[-1] = 0.0

        # these are the different attitude models that are used.
        # these can be put into a function to just call the one that is
        # currently being used: this is set in the input to the run
        # game function with a string

        # for speed, the model computes the attiutdes at the beginning
        # of each game rather than each trial. For some games where
        # within round learning is important, this could be changed

        if "tree_rocks" in attitude_condition:
            if epoch > 2:
                for i in range(world_size):
                    for j in range(world_size):
                        object_state = torch.tensor(
                            env.world[i, j, 0].appearance[:-3]
                        ).float()
                        predict = resource_model(object_state)
                        predict = predict.detach().numpy()

                        env.world[i, j, 0].appearance[-3] = predict[0] * 255
                        env.world[i, j, 0].appearance[-2] = predict[1] * 255
                        env.world[i, j, 0].appearance[-1] = predict[2] * 255
                        # if epoch % 100 == 0:
                        #    print(
                        #        "tree rocks",
                        #        epoch,
                        #        i,
                        #        j,
                        #        env.world[i, j, 0].appearance[-3:],
                        #        env.world[i, j, 0].kind,
                        #    )

        # --------------------------------------------------------------
        # this model creates a neural network to learn the reward values
        # --------------------------------------------------------------

        if "implicit_attitude" in attitude_condition:
            if epoch > 2:
                for i in range(world_size):
                    for j in range(world_size):
                        object_state = torch.tensor(
                            env.world[i, j, 0].appearance[:-3]
                        ).float()
                        # object_state = torch.concat(
                        #    object_state, agent_wood, agent_stone
                        # )
                        rs, _ = value_model(object_state.unsqueeze(0))
                        r = rs[0][1]
                        env.world[i, j, 0].appearance[-2] = r.item() * 255
            testing = False
            if testing and epoch % 100 == 0:
                atts = eval_attiude_model()
                print(epoch, atts)

        # --------------------------------------------------------------
        # this is the no attitude condition, simple IQN learning
        # --------------------------------------------------------------

        if (
            attitude_condition == "no_attitude"
        ):  # this sets a control condition where no attitudes are used
            for i in range(world_size):
                for j in range(world_size):
                    env.world[i, j, 0].appearance[-2] = 0.0

        # --------------------------------------------------------------
        # this is our episodic memory model with search and weighting
        # --------------------------------------------------------------

        if (
            "EWA" in attitude_condition and epoch > 100
        ):  # this sets a control condition where no attitudes are used
            object_memory_states_tensor = torch.tensor(
                [obj_mem[0] for obj_mem in object_memory]
            )
            full_view = False
            if full_view:
                for i in range(world_size):
                    for j in range(world_size):
                        o_state = env.world[i, j, 0].appearance[:-3]
                        mems = k_most_similar_recent_states(
                            torch.tensor(o_state),
                            state_knn,
                            object_memory,
                            object_memory_states_tensor,
                            decay_rate=1.0,
                            k=100,
                        )
                        env.world[i, j, 0].appearance[-1] = (
                            compute_weighted_average(
                                o_state,
                                mems,
                                similarity_decay_rate=similarity_decay_rate,
                                time_decay_rate=episodic_decay_rate,
                            )
                            * 255
                        )
            else:
                for i in range(9):
                    for j in range(9):
                        if (
                            i - 4 >= 0
                            and j - 4 >= 0
                            and i + 4 < world_size
                            and j + 4 < world_size
                        ):
                            o_state = env.world[i - 4, j - 4, 0].appearance[:-3]
                            mems = k_most_similar_recent_states(
                                torch.tensor(o_state),
                                state_knn,
                                object_memory,
                                object_memory_states_tensor,
                                decay_rate=1.0,
                                k=10,
                            )
                            env.world[i, j, 0].appearance[-1] = (
                                compute_weighted_average(
                                    o_state,
                                    mems,
                                    similarity_decay_rate=similarity_decay_rate,
                                    time_decay_rate=episodic_decay_rate,
                                )
                                * 255
                            )

        # --------------------------------------------------------------
        # this is complementary learning system model
        # --------------------------------------------------------------

        if (
            "CMS" in attitude_condition and epoch > 100
        ):  # this sets a control condition where no attitudes are used
            object_memory_states_tensor = torch.tensor(
                [obj_mem[0] for obj_mem in object_memory]
            )
            for i in range(world_size):
                for j in range(world_size):
                    o_state = env.world[i, j, 0].appearance[:-3]
                    mems = k_most_similar_recent_states(
                        torch.tensor(o_state),
                        state_knn_CMS,  # HERE IS THE ERROR!
                        object_exp2,
                        object_memory_states_tensor,
                        decay_rate=1.0,
                        k=100,
                    )
                    env.world[i, j, 0].appearance[-1] = (
                        compute_weighted_average(
                            o_state,
                            mems,
                            similarity_decay_rate=similarity_decay_rate,
                            time_decay_rate=episodic_decay_rate,
                        )
                        * 255
                    )

        turn = 0

        start_time = time.time()

        while done == 0:
            """
            Find the agents and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            # --------------------------------------------------------------
            # note the sync models lines may need to be deleted
            # the IQN has a soft update, so we should test dropping
            # the lines below

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for mods in trainable_models:
                    models[mods].qnetwork_target.load_state_dict(
                        models[mods].qnetwork_local.state_dict()
                    )
            # --------------------------------------------------------------

            agentList = find_instance(env.world, "neural_network")

            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0
            agent_num = 0

            for loc in agentList:
                agent_num = agent_num + 1
                if env.world[loc].kind != "deadAgent":
                    holdObject = env.world[loc]
                    device = models[holdObject.policy].device

                    # state is 1,1,11,9,9 [need to get wood and stone into state space]
                    state = env.pov(loc)
                    num_wood = torch.tensor(holdObject.wood).float()
                    num_stone = torch.tensor(holdObject.stone).float()

                    batch, timesteps, channels, height, width = state.shape

                    action = models[env.world[loc].policy].take_action(state, epsilon)

                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                        object_info,
                        resource_outcome,
                    ) = holdObject.transition(env, models, action[0], loc)

                    # --------------------------------------------------------------
                    # create object memory
                    # this sets up the direct reward experience and state information
                    # to be saved in a replay and also learned from

                    state_object = object_info[0:-3]
                    state_object_input = torch.tensor(state_object).float()

                    rs, _ = value_model(state_object_input.unsqueeze(0))
                    if epoch < 100:
                        mem = (state_object, reward)
                        object_exp2.append(mem)
                    else:
                        if reward > torch.max(rs) or reward < torch.min(rs):
                            mem = (state_object, reward)
                            object_exp2.append(mem)

                    object_exp = (state_object, reward)
                    value_model.add_memory(state_object, reward)

                    # learn resource of target
                    if reward != 0:
                        resource_model.add_memory(state_object, resource_outcome)
                    else:
                        if random.random() > 0.5:  # seems to work if downsample nothing
                            resource_model.add_memory(state_object, resource_outcome)

                    if len(resource_model.replay_buffer) > 33 and turn % 2 == 0:
                        resource_loss = resource_model.learn(
                            resource_model.sample(32), batch_size=32
                        )
                    # if epoch > 100 and epoch % 40 and turn % 2:
                    #    if resource_outcome != [1, 0, 0]:
                    #        predict = resource_model(torch.tensor(state_object).float())
                    #        print(predict, resource_outcome)

                    # note, to save time we can toggle the line below to only learn the
                    # implicit attitude when the implicit attitude is being used.

                    if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                        memories = value_model.sample(50)
                        value_loss = value_model.learn(memories, 25)
                    object_memory.append(object_exp)
                    if USE_KNN_MODEL:
                        # Fit a k-NN model to states extracted from the replay buffer
                        state_knn.fit([exp[0] for exp in object_memory])
                        state_knn_CMS.fit([exp[0] for exp in object_exp2])

                    # --------------------------------------------------------------
                    reward_values = env.gem_values
                    reward_values = sorted(reward_values, reverse=True)

                    if reward == 10:
                        gems[0] = gems[0] + 1
                    if reward == -5:
                        gems[1] = gems[1] + 1
                    if reward == 0:
                        gems[2] = gems[2] + 1
                    if reward == -1:
                        gems[3] = gems[3] + 1

                    # note, the line for PER is commented out. We may want to use IQN
                    # with PER as a better comparison

                    done_flag = 0
                    if (
                        withinturn > max_turns
                        or len(find_instance(env.world, "neural_network")) == 0
                    ):
                        done_flag = 1
                    exp = (
                        # models[env.world[new_loc].policy].max_priority,
                        1,
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done_flag,
                        ),
                    )

                    env.world[new_loc].episode_memory.append(exp)

                    if env.world[new_loc].kind == "agent":
                        # game_points[0] = game_points[0] + reward / reward_values[0]
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

            if epoch > 10 and withinturn % modelUpdate_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for mods in trainable_models:
                    experiences = models[mods].memory.sample()
                    loss = models[mods].learn(experiences)
                    losses = losses + loss

        if epoch > 10:
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

        if epoch % 20 == 0 and len(trainable_models) > 0 and epoch != 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),
                gems,
                losses,
                epsilon,
                str(gem_changes),
                attitude_condition,
                # env.gem1_value,
                # env.gem2_value,
                # env.gem3_value,
            )
            # rs = show_weighted_averaged(object_memory)
            # print(epoch, rs)
            game_points = [0, 0]
            gems = [0, 0, 0, 0]
            losses = 0
    return models, env, turn, epsilon


models = create_models()

# options here are. these are experiments that we ran

run_params = (
    [0.5, 4010, 20, 0.999, "EWA", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "tree_rocks", 12000, 2500, 20.0, 20.0],
    [0.5, 4010, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    [0.5, 4010, 20, 0.999, "None", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "implicit_attitude", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "CMS", 12000, 2500, 20.0, 20.0],
    # [0.5, 4010, 20, 0.999, "EWA", 12000, 2500, 20.0, 20.0],
)


# Convert the tuple of lists to a list of lists
# run_params_list = list(run_params)

# Shuffle the list of lists
# random.shuffle(run_params_list)

# If you need the result as a tuple again
# run_params_list = tuple(run_params_list)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models = create_models()
    value_model = ValueModel(state_dim=8, memory_size=250)
    resource_model = ResourceModel(state_dim=8, memory_size=2000)
    object_memory = deque(maxlen=run_params[modRun][6])
    state_knn = NearestNeighbors(n_neighbors=5)
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
        epsilon_decay=run_params[modRun][3],
        attitude_condition=run_params[modRun][4],
        switch_epoch=run_params[modRun][5],
        episodic_decay_rate=run_params[modRun][7],
        similarity_decay_rate=run_params[modRun][8],
    )
    # atts = eval_attiude_model()
    # print(atts)


# notes:
#      retreived memorories can be put back into a replay buffer
#      need to have long term memories that get stored somehow
#      if we can get the decay to work right, decay can be something that
#      is modulated (and maybe learned) to retain memories for longer
