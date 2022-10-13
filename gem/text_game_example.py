# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
from gem.environment.elements.taxiCab_elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)
from models.linear_lstm_dqn_PER import Model_linear_LSTM_DQN
from gemworld.taxiCab import TaxiCabEnv
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from DQN_utils import get_TD_error, save_models, load_models, make_video
import torch

import random

# save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"
# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

print(device)

def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.001,
            replay_size=1024,  # 2048
            in_size=4,  # 650
            hid_size1=10,  # 75
            hid_size2=10,  # 30
            out_size=2,
            priority_replay=False,
            device=device,
        )
    )  # taxi model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
        models[model].model2.to(device)

    return models


models = create_models()


# test game area


input = torch.tensor([1.,2.,1.,2.]).unsqueeze(0).to(device)
out = models[0].model1(input)
print(out)


import numpy as np
sm = nn.Softmax(dim=1)

def generate_alien():
    alien_type = np.random.choice([0,1])
    if alien_type == 0:
        appearence = [alien_type, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
        cooperation = np.random.choice([-1,1], p = (.1, .9))
    if alien_type == 1:
        appearence = [alien_type, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
        cooperation = np.random.choice([-1,1], p = (.9, .1))
    return alien_type, appearence, cooperation

from collections import deque

class Agent():
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [0.0, 0.0, 255.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.passable = 0  # whether the object blocks movement
        self.trainable = 1  # whether there is a network to be optimized
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"

agent = Agent(0)

max_turns = 100
modelUpdate_freq = 25
turn = 0
done = 0
approaches = [0,0]
losses = 0

for epoch in range(10000000):
    alien_type, appearence, cooperation = generate_alien()
    appearence = torch.tensor(appearence).float().unsqueeze(0).to(device)

    action = models[0].take_action([appearence, .1])
    state = appearence
    next_state = appearence
    if action == 0:
        reward = 0
    if action == 1:
        reward = cooperation

    approaches[alien_type] = approaches[alien_type] + action

    if turn == max_turns:
        done = True

    exp = [1, (
        state,
        action,
        reward,
        next_state,
        done,
    )]

    agent.episode_memory.append(exp)

    exp = (
        exp[0],
        (
            exp[1][0].to(device),
            torch.tensor(exp[1][1]).float().to(device),
            torch.tensor(exp[1][2]).float().to(device),
            exp[1][3].to(device),
            torch.tensor(exp[1][4]).float().to(device),
            ),
        )

    models[0].PER_replay.add(exp[0], exp[1])
         
    if epoch % modelUpdate_freq == 0:
        loss = models[0].training(256, 0.9)
        losses = losses + loss.detach().cpu().numpy()

    if epoch % 5000 == 0:
        print("epoch:" , epoch, "loss: ",losses/5000, "approaches: ", approaches)
        approaches = [0,0]
        losses = 0




