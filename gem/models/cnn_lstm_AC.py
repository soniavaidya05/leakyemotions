from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.memory import Memory
from gem.models.perception import agent_visualfield


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
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        n_layers=1,
        batch_first=True,
    ):
        super(Combine_CLD_AC, self).__init__()
        self.cnn = CNN_CLD(numFilters)
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hid_size1, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)

        self.actor_lin1 = nn.Linear(hid_size2, out_size)
        self.l3 = nn.Linear(hid_size2, 25)
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


class Model_CNN_LSTM_AC:

    kind = "cnn_lstm_AC"  # class variable shared by all instances

    def __init__(
        self, numFilters, lr, replay_size, in_size, hid_size1, hid_size2, out_size
    ):
        self.modeltype = "cnn_lstm_ac"
        self.model1 = Combine_CLD_AC(
            numFilters, in_size, hid_size1, hid_size2, out_size
        )
        self.model2 = Combine_CLD_AC(
            numFilters, in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.replay = deque([], maxlen=replay_size)
        self.sm = nn.Softmax(dim=1)
        self.values = []
        self.logprobs = []
        self.rewards = []

    def createInput(self, world, i, j, holdObject, numMemories=-1):
        """
        Creates a single input in RNN format
        """

        img = agent_visualfield(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = input.unsqueeze(0)
        return input

    def createInput2(self, world, i, j, holdObject, seqLength=-1):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: (1) need to get createInput and createInput2 into one function
        TODO: (2) test whether version 1 and version 2 below work properly
              Specifically, whether the sequences in version 2 are being
              stacked properly
        """

        img = agent_visualfield(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = input.unsqueeze(0)
        if seqLength == -1:
            combined_input = input

        version = 2

        if version == 1:
            if seqLength > 1:
                previous_input1 = world[i, j, 0].replay[-2][0]
                previous_input2 = world[i, j, 0].replay[-1][0]
                combined_input = torch.cat(
                    [previous_input1, previous_input2, input], dim=1
                )

        if version == 2:
            # will the code below work?
            if seqLength > 1:
                combined_input = input
                for mem in range(seqLength):
                    previous_input = world[i, j, 0].replay[(mem + 1) * -1][0]
                    combined_input = torch.cat([previous_input, combined_input], dim=1)

        return input, combined_input

    def pov(self, world, location, holdObject, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        TODO: to get better flexibility, this code should be moved to env
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        state_now = torch.tensor([])
        for layer in layers:
            """
            Loops through each layer to get full visual field
            """
            loc = (location[0], location[1], layer)
            img = agent_visualfield(world, loc, holdObject.vision)
            input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
            state_now = torch.cat((state_now, input.unsqueeze(0)), dim=2)

        if len(inventory) > 0:
            """
            Loops through each additional piece of information and places into one layer
            """
            inventory_var = torch.tensor([])
            for item in range(len(inventory)):
                tmp = (current_state[:, -1, -1, :, :] * 0) + inventory[item]
                inventory_var = torch.cat((inventory_var, tmp), dim=0)
            inventory_var = inventory_var.unsqueeze(0).unsqueeze(0)
            state_now = torch.cat((state_now, inventory_var), dim=2)

        current_state[:, -1, :, :, :] = state_now

        return current_state

    def take_action(self, inp):
        """
        Takes action from the input
        TODO: different models use different input parameters
              need to figure out how to standardize
        """

        # inp, epsilon = params
        policy, value = self.model1(inp)
        # self.values.append(value)

        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        # self.logprobs.append(logprob_)
        return action, logprob_, value

    def training(self, batch_size=100, clc=0.1):
        """
        Actor critic training. This works but as an unncessary batch_size
        Input that we should figure out if there is a way to make more efficient
        """
        loss = torch.tensor(0.0)
        if len(self.values) > 1:
            actor_loss = -1 * self.logprobs * (self.Returns - self.values.detach())
            critic_loss = torch.pow(self.values - self.Returns, 2)
            loss = actor_loss.sum() + clc * critic_loss.sum()
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        """
        Update double DQN model
        """
        pass

    def transfer_memories(self, world, loc, extra_reward=True, seqLength=5):
        """
        Transfer the indiviu=dual memories to the model
        TODO: A second function is written below because the inputs for
              Actor-Critic and DQN are different. Need to figure out how to
              code that one generic set of functions will work
        """
        pass

    def transfer_memories_AC(self, world, loc):
        """
        Transfer the indiviu=dual memories to the model
        TODO: A second function is written below because the inputs for
              Actor-Critic and DQN are different. Need to figure out how to
              code that one generic set of functions will work
        TODO: Gamma should be passed into this model rather than being predefined
        """
        rewards = world[loc].AC_reward.flip(dims=(0,)).view(-1)
        logprobs = world[loc].AC_logprob.flip(dims=(0,)).view(-1)
        values = world[loc].AC_value.flip(dims=(0,)).view(-1)

        gamma = 0.8
        # gamma = 0.95
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]):  # B
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        # Returns = F.normalize(Returns, dim=0)

        self.rewards = torch.concat([self.rewards, rewards])
        self.values = torch.concat([self.values, values])
        self.logprobs = torch.concat([self.logprobs, logprobs])
        self.Returns = torch.concat([self.Returns, Returns])
