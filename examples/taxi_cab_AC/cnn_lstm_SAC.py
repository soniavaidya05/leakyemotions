import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal

# import gym


# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/buffer.py


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.state_memory = torch.zeros(
            [self.mem_size, 4, 9, 9]
        )  # will need to not have this hard coded
        # self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = torch.zeros(
            [self.mem_size, 4, 9, 9]
        )  # will need to not have this hard coded
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CNN_CLD(nn.Module):
    def __init__(self, in_channels, numFilters):
        super(CNN_CLD, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels, out_channels=numFilters, kernel_size=1
        )
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)

    def forward(self, x):
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        # print(y.shape)

        y = y.flatten()
        elements = y.shape[0]
        y = y.reshape(int(elements / 650), 650)

        return y


import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="critic",
        chkpt_dir="tmp/sac",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.cnn = CNN_CLD(4, 5)  # update this to be parameters once working

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        state_flatten = self.cnn(state)
        action_value = self.fc1(T.cat([state_flatten, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="value",
        chkpt_dir="tmp/sac",
    ):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.cnn = CNN_CLD(4, 5)  # update this to be parameters once working

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        state_value = self.cnn(state)
        state_value = self.fc1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        max_action,
        fc1_dims=256,
        fc2_dims=256,
        n_actions=4,
        name="actor",
        chkpt_dir="tmp/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.cnn = CNN_CLD(4, 5)  # update this to be parameters once working

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.cnn(state)
        prob = self.fc1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


import os
import torch as T
import torch.nn.functional as F
import numpy as np

# from buffer import ReplayBuffer
# from networks import ActorNetwork, CriticNetwork, ValueNetwork


class SAC:
    # def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
    #        env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
    #        layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[650],
        env=None,
        gamma=0.99,
        n_actions=4,
        max_size=1000000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=128,  # turn this back to 256?
        reward_scale=2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
        #            name='actor', max_action=env.action_space.high)
        self.actor = ActorNetwork(
            alpha, input_dims, n_actions=n_actions, name="actor", max_action=1
        )  # I think this is right because the actions are binaries
        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.value = ValueNetwork(beta, input_dims, name="value")
        self.target_value = ValueNetwork(beta, input_dims, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print(".... saving models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
        return (
            critic_loss.detach().cpu().numpy(),
            actor_loss.detach().cpu().numpy(),
            value_loss.detach().cpu().numpy(),
        )


model = SAC()

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
from examples.taxi_cab.elements import (
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
save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

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

from examples.taxi_cab_AC.cnn_lstm_AC import Model_CNN_LSTM_AC


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_AC(
            in_channels=4,
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


models = create_models()
epochs = 1000000
max_turns = 100


done = 0
turn = 0
losses = [0, 0, 0]

for epoch in range(epochs):
    done = 0
    turn = 0

    env.reset_env(height=world_size, width=world_size, layers=1)

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(3)
        env.world[loc].reward = 0

    while done == 0:

        turn = turn + 1
        agentList = find_instance(env.world, "neural_network")
        random.shuffle(agentList)

        for loc in agentList:
            if env.world[loc].action_type == "neural_network":

                holdObject = env.world[loc]
                device = "cpu"

                state = env.pov(loc, inventory=[holdObject.has_passenger], layers=[0])
                # get rid of LSTM for now
                state = state.squeeze()[-1, :, :, :]
                actions, _ = model.actor.sample_normal(state, reparameterize=False)
                action_dist = torch.distributions.Categorical(logits=actions)
                action = action_dist.sample()  # E
                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                ) = holdObject.transition(env, models, action, loc)

                game_points[0] = game_points[0] + reward
                if reward > 10:
                    game_points[1] = game_points[1] + 1

                # get rid of LSTM for now
                next_state = next_state.squeeze()[-1, :, :, :]
                action_state = [0, 0, 0, 0]
                action_state[action] = 1
                action = torch.tensor(action_state)
                model.remember(state, action, reward, next_state, done)
                

        if turn == max_turns:
            done = 1

    loss = model.learn()

    if epoch % 50 == 0:
        print(
            "epoch: ",
            epoch,
            "losses: ",
            loss,
            game_points,
        )
        game_points = [0, 0]
        losses = [0, 0, 0]
