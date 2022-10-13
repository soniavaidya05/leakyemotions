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


class SIMPLE_DQN(nn.Module):
    """
    TODO: need to be able to have an input for non CNN layers to add additional inputs to the model
            likely requires an MLP before the LSTM where the CNN and the additional
            inputs are concatenated
    """

    def __init__(
        self,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        n_layers=2,
        batch_first=True,
    ):
        super(SIMPLE_DQN, self).__init__()
        self.l1 = nn.Linear(in_size, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.l3 = nn.Linear(hid_size2, out_size)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        """
        TODO: check the shapes below. 
        """
        y = F.relu(self.l1(x))
        #print("y.shape: ", y.shape)
        y = F.relu(self.l2(y))
        y = self.l3(y)

        return y


class Model_simple_linear_DQN:

    kind = "linear_dqn"  # class variable shared by all instances

    def __init__(
        self,
        lr,
        replay_size,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        priority_replay=True,
        device="cpu",
    ):
        self.modeltype = "linear_dqn"
        self.model1 = SIMPLE_DQN(
            in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.sm = nn.Softmax(dim=1)
        self.device = device

    def softmax(self, x):
        return(np.exp(x)/np.exp(x).sum())


    def take_action(self, params):
        """
        Takes action from the input
        """

        inp, epsilon = params
        Q = self.model1(inp)
        vals = Q.cpu().detach().numpy()
        p = self.softmax(vals)
        #p = self.sm(Q).cpu().detach().numpy()

        if epsilon > 0.3:
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().cpu().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p=np.asarray(p))
        return action

    def training(self, exp):

        priority, (state, action, reward, next_state, done) = exp

        Q1 = self.model1(state)
        Y = torch.tensor(reward).float().to(self.device)
        X = Q1[action]
        loss = self.loss_fn(X, Y)
        loss.backward()
        self.optimizer.step()

        return loss


