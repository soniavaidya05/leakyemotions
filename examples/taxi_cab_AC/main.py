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
from examples.taxi_cab_AC.cnn_lstm_SAC import Model_CNN_LSTM_AC
from  examples.taxi_cab_AC.env import TaxiCabEnv
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
save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

device = "cpu"
print(device)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_AC(
            in_channels = 4,
            numFilters = 5, 
            lr = .001, 
            replay_size = 3, 
            in_size = 650, 
            hid_size1 = 75, 
            hid_size2 = 30, 
            out_size = 4
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


# building area
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




for epoch in range(epochs):
    done = 0
    turn = 0

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(3)

        env.world[loc].AC_reward = torch.tensor([])
        env.world[loc].AC_values = torch.tensor([])
        env.world[loc].AC_logprobs = torch.tensor([])

        models[0].values = torch.tensor([])
        models[0].logprobs = torch.tensor([])
        models[0].rewards = torch.tensor([])
        models[0].Returns = torch.tensor([])

        #env.world[loc].Returns = []


    while done == 0:
        turn = turn + 1


        agentList = find_instance(env.world, "neural_network")
        random.shuffle(agentList)

        loc = agentList[0]

        env.world[loc].reward = 0

        if env.world[loc].kind != "deadAgent":

            (
                state,
                action,
                reward,
                next_state,
                done,
                new_loc,
                info,
            ) = env.step_AC(models, loc, epsilon)

        logprob, value = info
        env.world[new_loc].AC_reward = torch.cat((env.world[new_loc].AC_reward, torch.tensor(reward).float().reshape(1,1)),0)
        env.world[new_loc].AC_values = torch.cat((env.world[new_loc].AC_values, value),0)
        env.world[new_loc].AC_logprobs = torch.cat((env.world[new_loc].AC_logprobs, logprob.reshape(1,1)),0)

        game_points[0] = game_points[0] + reward
        if reward > 10:
            game_points[1] =game_points[1] + 1

        if turn == max_turns:
            done = 1


    loc = find_instance(env.world, "neural_network")[0]

    models[0].transfer_memories(env.world, loc)
    loss = models[0].training()

    if epoch % 500 == 0:
        # print the state and update the counters. This should be made to be tensorboard instead
        print(
            epoch,
            turn,
            round(game_points[0]),
            round(game_points[1]),
            losses,
            epsilon,
            world_size,
        )
        game_points = [0, 0]
        losses = 0


