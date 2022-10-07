from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.memory import Memory
from models.perception import agent_visualfield


import random
import numpy as np
from collections import deque

from models.priority_replay import Memory, SumTree


class CNN_CLD(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(CNN_CLD, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=1
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


class Combine_CLD(nn.Module):
    def __init__(
        self,
        in_channels,
        num_filters,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        n_layers=2,
        batch_first=True,
    ):
        super(Combine_CLD, self).__init__()
        self.cnn = CNN_CLD(in_channels, num_filters)
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hid_size1, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.l3 = nn.Linear(hid_size2, out_size)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        r_out, (h_n, h_c) = self.rnn(r_in)

        y = F.relu(self.l1(r_out[:, -1, :]))
        y = F.relu(self.l2(y))
        y = self.l3(y)

        return y


class Model_CNN_LSTM_DQN:

    kind = "cnn_lstm_dqn"  # class variable shared by all instances

    def __init__(
        self,
        in_channels,
        num_filters,
        lr,
        replay_size,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        priority_replay=True,
        device="cpu",
    ):
        self.modeltype = "cnn_lstm_dqn"
        self.model1 = Combine_CLD(
            in_channels, num_filters, in_size, hid_size1, hid_size2, out_size
        )
        self.model2 = Combine_CLD(
            in_channels, num_filters, in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.sm = nn.Softmax(dim=1)
        self.priority_replay = priority_replay
        if priority_replay == True:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0.06,  # set this to 0 for uniform sampling (check these numbers)
                beta=0.4,  # set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0.0001,  # set this to 0 for uniform sampling (check these numbers)
            )
        if priority_replay == False:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0,  # set this to 0 for uniform sampling (check these numbers)
                beta=0,  # set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0,  # set this to 0 for uniform sampling (check these numbers)
            )
        self.device = device

    def pov(self, world, location, holdObject, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
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

    def take_action(self, params):
        """
        Takes action from the input
        """

        inp, epsilon = params
        Q = self.model1(inp)
        p = self.sm(Q).detach().numpy()[0]

        if epsilon > 0.3:
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p=p)
        return action

    def training(self, batch_size, gamma):
        """
        DQN batch learning
        """

        loss = torch.tensor(0.0)

        # note, there may be a ratio of priority replay to random replay that could be ideal
        # need to figure out how to get current_replay_size
        current_replay_size = batch_size + 1

        if current_replay_size > batch_size:
            # if self.priority_replay == False:
            #    # this needs to be rewritten to work without priority replay
            #    minibatch = random.sample(self.replay, batch_size)
            # if self.priority_replay == True:
            minibatch, idxs, is_weight = self.PER_replay.sample(batch_size)

            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            Q1 = self.model1(state1_batch)
            with torch.no_grad():
                Q2 = self.model2(state2_batch)

            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            errors = torch.abs(Y - X).data.numpy()

            # there should be better ways of doing the following
            self.max_priority = np.max(errors)

            # update priority
            for i in range(len(errors)):
                idx = idxs[i]
                self.PER_replay.update(idx, errors[i])

            self.optimizer.zero_grad()
            if self.priority_replay == False:
                loss = self.loss_fn(X, Y.detach())
            if self.priority_replay == True:
                replay_stable = 0
                if replay_stable == 1:
                    loss = self.loss_fn(X, Y.detach())
                if replay_stable == 0:
                    loss = (torch.FloatTensor(is_weight) * F.mse_loss(Y, X)).mean()
                    # compute this twice!
                    # loss = (
                    #    torch.FloatTensor(is_weight)
                    #    * ((X - Y.detach()) ** 2)
                    # ).mean()
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        """
        Update double DQN model
        """
        self.model2.load_state_dict(self.model1.state_dict())

    def transfer_memories(self, world, loc, extra_reward=True, seqLength=4):
        """
        Transfer the indiviu=dual memories to the model
        TODO: We need to have a single version that works for both DQN and
              Actor-criric models (or other types as well)
        """
        exp = world[loc].episode_memory[-1]
        high_reward = exp[1][2]
        # move experience to the gpu if available
        exp = (
            torch.tensor(exp[0]).to(self.device),
            (
                exp[1][0].to(self.device),
                torch.tensor(exp[1][1]).to(self.device),
                torch.tensor(exp[1][2]).to(self.device),
                exp[1][3].to(self.device),
                torch.tensor(exp[1][4]).to(self.device),
            ),
        )
        self.PER_replay.add(exp[0], exp[1])
        if extra_reward == True and abs(high_reward) > 9:
            for _ in range(seqLength):
                self.PER_replay.add(exp[0], exp[1])
