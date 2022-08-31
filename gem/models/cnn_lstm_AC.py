from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.memory import Memory
from models.perception import agentVisualField


class CNN_CLD(nn.Module):
    def __init__(self, numFilters):
        super(CNN_CLD, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=numFilters, kernel_size=1
        )
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)

    def forward(self, x):
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        return y


class Combine_CLD_AC(nn.Module):
    def __init__(
        self,
        numFilters,
        insize,
        hidsize1,
        hidsize2,
        outsize,
        n_layers=1,
        batch_first=True,
    ):
        super(Combine_CLD_AC, self).__init__()
        self.cnn = CNN_CLD(numFilters)
        self.rnn = nn.LSTM(
            input_size=insize,
            hidden_size=hidsize1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hidsize1, hidsize1)
        self.l2 = nn.Linear(hidsize1, hidsize2)

        self.actor_lin1 = nn.Linear(hidsize2, outsize)
        self.l3 = nn.Linear(hidsize2, 25)
        self.critic_lin1 = nn.Linear(25, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        r_out, (h_n, h_c) = self.rnn(r_in)
        y = F.leaky_relu(self.l1(r_out[:, -1, :]))
        # y = self.dropout(y)
        y = F.leaky_relu(self.l2(y))
        # y = self.dropout(y)
        # need to check the dim below, changed from 0 to -1
        # actor = self.actor_lin1(y)
        # actor = nn.LogSoftmax(self.actor_lin1(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=-1)
        # actor = F.log_softmax(self.actor_lin1(y), dim=1)
        # actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.leaky_relu(self.l3(y.detach()))
        # critic = torch.tanh(self.critic_lin1(c))
        # why is the tanh in some of the examples?
        critic = self.critic_lin1(c)

        return actor, critic


class model_CNN_LSTM_AC:

    kind = "cnn_lstm_AC"  # class variable shared by all instances

    def __init__(self, numFilters, lr, replaySize, insize, hidsize1, hidsize2, outsize):
        self.modeltype = "cnn_lstm_ac"
        self.model1 = Combine_CLD_AC(numFilters, insize, hidsize1, hidsize2, outsize)
        self.model2 = Combine_CLD_AC(numFilters, insize, hidsize1, hidsize2, outsize)
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.replay = deque([], maxlen=replaySize)
        self.sm = nn.Softmax(dim=1)
        self.values = []
        self.logprobs = []
        self.rewards = []

    def createInput(self, world, i, j, holdObject, numMemories=-1):
        # note, this is not actually an LSTM for this model
        # because it just uses the current image.
        # will need to update to use the memories tag to create an image
        # from the replay. But, still need to figure out the
        # one vs. two outputs from these functions

        img = agentVisualField(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = input.unsqueeze(0)
        return input

    def takeAction(self, inp):
        # inp, epsilon = params
        policy, value = self.model1(inp)
        # self.values.append(value)

        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        # self.logprobs.append(logprob_)
        return action, logprob_, value

    def training(self, batch_size, gamma):
        clc = 0.1
        gamma = 0.95
        rewards = torch.Tensor(self.rewards).flip(dims=(0,)).view(-1)  # A
        logprobs = torch.stack(self.logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(self.values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]):  # B
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns, dim=0)
        actor_loss = -1 * logprobs * (Returns - values.detach())  # C
        critic_loss = torch.pow(values - Returns, 2)  # D
        loss = actor_loss.sum() + clc * critic_loss.sum()  # E
        loss.backward()
        self.optimizer.step()
        return loss

    def updateQ(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def transferMemories(self, world, i, j, extraReward=True, seqLength=5):
        pass
        # reward = 0  # needs to be added into this command
        # self.rewards.append(reward)

    def transferMemories_AC(self, reward):
        reward = 0  # needs to be added into this command
        self.rewards.append(reward)
