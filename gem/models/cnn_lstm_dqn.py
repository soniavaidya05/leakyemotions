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


class Combine_CLD(nn.Module):
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
        super(Combine_CLD, self).__init__()
        self.cnn = CNN_CLD(numFilters)
        self.rnn = nn.LSTM(
            input_size=insize,
            hidden_size=hidsize1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hidsize1, hidsize1)
        self.l2 = nn.Linear(hidsize1, hidsize2)
        self.l3 = nn.Linear(hidsize2, outsize)

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


class model_CNN_LSTM_DQN:

    kind = "cnn_lstm_dqn"  # class variable shared by all instances

    def __init__(self, numFilters, lr, replaySize, insize, hidsize1, hidsize2, outsize):
        self.modeltype = "cnn_lstm_dqn"
        self.model1 = Combine_CLD(numFilters, insize, hidsize1, hidsize2, outsize)
        self.model2 = Combine_CLD(numFilters, insize, hidsize1, hidsize2, outsize)
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.replay = deque([], maxlen=replaySize)
        self.sm = nn.Softmax(dim=1)

    def createInput(self, world, i, j, holdObject, numMemories = -1):
        if numMemories > 0:
            for _ in range(numMemories):
                print("Do Something")
        img = agentVisualField(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = input.unsqueeze(0)
        return input

    def takeAction(self, params):
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

        loss = torch.tensor(0.0)

        if len(self.replay) > batch_size:

            minibatch = random.sample(self.replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            # next test, remove the reshape and see if this still works
            # Q1 = self.model1(state1_batch.reshape(batch_size,3,9,9))
            Q1 = self.model1(state1_batch)
            with torch.no_grad():
                # Q2 = self.model2(state2_batch.reshape(batch_size,3,9,9))
                Q2 = self.model2(state2_batch)

            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            self.optimizer.zero_grad()
            loss = self.loss_fn(X, Y.detach())
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        self.model2.load_state_dict(self.model1.state_dict())
