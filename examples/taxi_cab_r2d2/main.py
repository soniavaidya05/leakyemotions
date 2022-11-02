# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
from examples.taxi_cab_r2d2.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)

from examples.taxi_cab_r2d2.model import Network

from gem.models.cnn_lstm_dqn_PER import Model_CNN_LSTM_DQN
from examples.taxi_cab_r2d2.env import TaxiCabEnv
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video
import torch

import random

# save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "/Users/ethan/gem_output/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

device = "cpu"
print(device)


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec

def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_DQN(
            in_channels=4,
            num_filters=5,
            lr=0.001,
            replay_size=1024,  # 2048
            in_size=650,  # 650
            hid_size1=75,  # 75
            hid_size2=30,  # 30
            out_size=4,
            priority_replay=False,
            device=device,
        )
    )  # taxi model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
        models[model].model2.to(device)

    return models


world_size = 10

trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

models = create_models()
env = TaxiCabEnv(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject,
)

env.game_test()


# -------------------------------------------------------------------------
# TESTING MODEL STRUCTURE AREA
# -------------------------------------------------------------------------

model = Network(
            action_dim = 4, 
            obs_shape=(4, 9, 9), 
            hidden_dim=100
            )

losses = 0
game_points = [0, 0]
done, withinturn = 0, 0

env.reset_env(
    height=world_size,
    width=world_size,
    layers=1,
)

for loc in find_instance(env.world, "neural_network"):
    # reset the memories for all agents
    # the parameter sets the length of the sequence for LSTM
    env.world[loc].init_replay(3)

agentList = find_instance(env.world, "neural_network")

from examples.taxi_cab_r2d2.model import AgentState

#empty_state = torch.zeros((1, 1, 4, 9, 9))
#state = AgentState(torch.from_numpy(obs).unsqueeze(0), 4)

loc = agentList[0]
env.world[loc].reward = 0

device = models[env.world[loc].policy].device
state = env.pov(loc, inventory=[env.world[loc].has_passenger], layers=[0])
state = state[:,-1,:,:,:]   # let's start with just a single input in
env.world[loc].state = AgentState(state, 4)


softMax = nn.Softmax()

epochs = 100

for epoch in range(epochs):

    agentList = find_instance(env.world, "neural_network")
    loc = agentList[0]
    

    state = env.pov(loc, inventory=[env.world[loc].has_passenger], layers=[0])
    state = state[:,-1,:,:,:]   # let's start with just a single input in
    if epoch > 0:
        env.world[loc].state.update(state.numpy().squeeze(0), env.world[loc].last_action, env.world[loc].last_reward, env.world[loc].hidden)

    q_value, recurrent_output = model(env.world[loc].state)

    probs = softMax(q_value).detach().numpy()
    action_int = np.random.choice([0,1,2,3], p = probs[0])

    action_one_hot = one_hot(4, action_int)

    (
        env.world,
        reward,
        next_state,
        done,
        new_loc,
    ) = env.world[loc].transition(env, models, action_int, loc)

    env.world[new_loc].last_action = action_int
    env.world[new_loc].last_reward = reward
    env.world[new_loc].hidden = recurrent_output

# only gotten this far


    self.local_buffer.add(action, reward, next_obs, q_value.numpy(), torch.cat(hidden).numpy())

    if done:
        block = self.local_buffer.finish()
        self.sample_queue.put(block)

    elif len(self.local_buffer) == self.block_length or episode_steps == self.max_episode_steps:
        with torch.no_grad():
            q_value, hidden = self.model(agent_state)

        block = self.local_buffer.finish(q_value.numpy())

        if self.epsilon > 0.01:
            block[2] = None
        self.sample_queue.put(block)

    if actor_steps % 400 == 0:
        self.update_weights()


    


















#x = model(state, prev_action, prev_reward, init_rnn_state)

next_rnn_state = None # the three below set up the first trial
prev_reward = torch.tensor(0.)
prev_action = torch.tensor([0,0,0,0])
pi, v, next_rnn_state = model(state, prev_action, prev_reward, next_rnn_state)

action_int = torch.argmax(pi)
action_one_hot = one_hot(4, action_int)

(
    env.world,
    reward,
    next_state,
    done,
    new_loc,
) = env.world[loc].transition(env, models, action_int, loc)

next_state = next_state[:,-1,:,:,:].unsqueeze(0)   # also converts to be just one time pesriod

# below is the replay buffer from our agent. We need to get into their format

exp = (
    models[env.world[new_loc].policy].max_priority,
    (
        state,
        action_one_hot,
        reward,
        next_state,
        done,
    ),
)

env.world[new_loc].episode_memory.append(exp)


# conceptually, the next forward pass would be (but, need to solve the replay and the training above)
state = next_state
prev_action = torch.tensor(action_one_hot).float()
prev_reward = torch.tensor(reward).float()
pi, v, next_rnn_state = model(state, prev_action, prev_reward, next_rnn_state)



# I found something called self.train() being called but can't find it






