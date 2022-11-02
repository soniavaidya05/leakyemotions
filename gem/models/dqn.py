from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.memory import Memory
from models.perception import agent_visualfield


class DQN(nn.Module):
    def __init__(self, numFilters, in_size, hid_size1, hid_size2, out_size):
        super(DQN, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=numFilters, kernel_size=1
        )
        self.l2 = nn.Linear(in_size, hid_size1)
        self.l3 = nn.Linear(hid_size1, hid_size1)
        self.l4 = nn.Linear(hid_size1, hid_size2)
        self.l5 = nn.Linear(hid_size2, out_size)
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)
        self.conv_bn = nn.BatchNorm2d(5)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        """
        forward of DQN
        """
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        y = F.relu(self.l2(y))
        # y = self.dropout(y)
        y = F.relu(self.l3(y))
        # y = self.dropout(y)
        y = F.relu(self.l4(y))
        # y = self.dropout(y)
        value = self.l5(y)
        return value


class ModelDQN:

    kind = "double_dqn"  # class variable shared by all instances

    def __init__(
        self, numFilters, lr, replay_size, in_size, hid_size1, hid_size2, out_size
    ):
        self.modeltype = "double_dqn"
        self.model1 = DQN(numFilters, in_size, hid_size1, hid_size2, out_size)
        self.model2 = DQN(numFilters, in_size, hid_size1, hid_size2, out_size)
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.replay = deque([], maxlen=replay_size)
        self.sm = nn.Softmax(dim=1)

    def createInput(self, world, i, j, holdObject, numImages=-1):
        """
        Creates a single input
        TODO: This is all that would be needed for a non-RNN model
              But we may need to consider changing this if a single function
              can handle both types of input files for RNN models
        """

        img = agent_visualfield(world, (i, j), holdObject.appearance.shape, k=holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        return input

    def take_action(self, params):
        """
        Selects an action
        TODO: right now params are read in. there may be a more efficient way of reading in
              kwargs
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
        Batch Double DQN learning
        """

        loss = torch.tensor(0.0)

        if len(self.replay) > batch_size:

            minibatch = random.sample(self.replay, batch_size)
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

            self.optimizer.zero_grad()
            loss = self.loss_fn(X, Y.detach())
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        """
        Update DQN model
        """
        self.model2.load_state_dict(self.model1.state_dict())

    def transfer_memories(self, world, loc, extra_reward=True):
        """
        Transfer the events from agent memory to model replay
        """

        exp = world[loc].replay[-1]
        self.replay.append(exp)
        if extra_reward == True and abs(exp[2]) > 9:
            for _ in range(5):
                self.replay.append(exp)
