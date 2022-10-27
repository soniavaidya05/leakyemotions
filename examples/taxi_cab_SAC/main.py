import math
import random
import numpy as np
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal



from examples.taxi_cab_SAC.toshikwa_scripts.model import BaseNetwork, DQNBase, QNetwork, TwinnedQNetwork, CategoricalPolicy, SAC


### TRY IN TAXI CAB

# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
from examples.taxi_cab_AC.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)

from examples.taxi_cab_AC.env import TaxiCabEnv
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video
import torch

import random

# save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "/Users/ethan/gem_output/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

device = "cpu"
print(device)

world_size = 10

trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

# models = create_models()
env = TaxiCabEnv(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject,
)

losses = 0
game_points = [0, 0]
epochs = 100
max_turns = 100
world_size = 10
env.reset_env(
    height=world_size,
    width=world_size,
    layers=1,
)

from gem.models.cnn_lstm_AC import Model_CNN_LSTM_AC



def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_AC(
            numFilters=5,
            lr=0.001,
            replay_size=3,
            in_size=650,
            hid_size1=75,
            hid_size2=30,
            out_size=4
            # note, need to add device and maybe in_channels
        )
    )  # taxi model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
        models[model].model2.to(device)
    # currently the AC has two models, which it doesn't need
    # and is wasting space

    return models

def explore(state):
    # Act with randomness.
    #state = torch.ByteTensor(
    #    state[None, ...]).to(device).float()
    with torch.no_grad():
        action, _, _ = model.policy.sample(state)
    return action.item()

def exploit(state):
    # Act without randomness.
    #state = torch.ByteTensor(
    #    state[None, ...]).to(device).float()
    with torch.no_grad():
        action = model.policy.act(state)
    return action.item()



models = create_models()
models[0].device = "cpu"
epochs = 1000000
max_turns = 100



losses = [0, 0, 0]
selected_actions = [0, 0, 0, 0]
m = nn.Softmax()


model = SAC(
        in_channels = 4,
        num_filters = 5,
        lr = .001,
        replay_size = 4096,
        in_size = 650,
        hid_size1 = 100,
        hid_size2= 75,
        out_size = 4,
        priority_replay=True,
        device="cpu",
        state_shape = (3,4,9,9),
        gamma = .9,
        use_per=True,
        num_steps = 1000000, # the next four are made up
        start_steps = 20000,
        update_interval = 10,
        multi_step = 1,
        target_entropy_ratio=0.98)


max_turns = 100

rewards = [0,0]
actions_taken = [0,0,0,0]

for epoch in range(epochs):

    env.reset_env(height=world_size, width=world_size, layers=1)
    turn = 0
    done = False
    while done == False:
        turn = turn + 1
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(3)
            env.world[loc].reward = 0



        turn = turn + 1
        agentList = find_instance(env.world, "neural_network")
        random.shuffle(agentList)

        holdObject = env.world[agentList[0]]

        state = env.pov(loc, inventory=[holdObject.has_passenger], layers=[0])
        #state = state.squeeze()[-1, :, :, :]
        #state = state.unsqueeze(0)
        action = explore(state)

        #if epoch < 20000:
        #    action = explore(state)
        #else:
        #    action = exploit(state)


        actions_taken[action] = actions_taken[action] + 1
        if turn == max_turns:
            done = True

        env.world,reward,next_state,done,new_loc = holdObject.transition(env, models, action, loc, done)
        #next_state = next_state.squeeze()[-1, :, :, :]
        #next_state = next_state[:, -1, :, :]


        model.memory.append(state, action, reward, next_state, done)  # real version has "clipped_reward" rather than reward
        rewards[0] = rewards[0] + reward
        if reward > .9:
            rewards[1] = rewards[1] + 1

        if epoch > 50 and turn % 5 == 0:
            model.learn()

        if epoch % 250:
            model.update_target()

    if epoch % 50 == 0 and epoch != 0:
        print(epoch, rewards, actions_taken)
        rewards = [0,0]
        actions_taken = [0,0,0,0]














