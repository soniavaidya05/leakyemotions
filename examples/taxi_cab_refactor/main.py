# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
from examples.taxi_cab.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)
from gem.models.cnn_lstm_dqn_PER import Model_CNN_LSTM_DQN
from examples.taxi_cab_refactor.env import TaxiCabEnv
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
loc = agentList[0]
env.world[loc].reward = 0

device = models[env.world[loc].policy].device
state = env.pov(loc, inventory=[env.world[loc].has_passenger], layers=[0])

state = state[:,-1,:,:,:].unsqueeze(0)   # let's start with just a single input in
from examples.taxi_cab_refactor.dqn_model import AtariLstmModel
model = AtariLstmModel(
            image_shape = state.shape[2:5],
            output_size = 4,
            fc_sizes=512,  # Between conv and lstm.
            lstm_size=512,
            use_maxpool=False,
            channels=[4 ,5],  # None uses default.
            kernel_sizes=[1,1],
            strides=[1,1],
            paddings=0,
            )


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

next_state = next_state[:,-1,:,:,:].unsqueeze(0)   # also converts to be just one time period

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






