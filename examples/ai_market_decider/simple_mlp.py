from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.memory import Memory
from gem.models.perception import agent_visualfield


import random
import numpy as np
from collections import deque

from gem.models.priority_replay import Memory, SumTree


class SIMPLE_MLP(nn.Module):
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
        super(SIMPLE_MLP, self).__init__()
        self.l1 = nn.Linear(in_size, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.adv = nn.Linear(hid_size2, out_size)
        self.value = nn.Linear(hid_size2, 1)

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        """
        TODO: check the shapes below. 
        """
        y = F.relu(self.l1(x))
        #print("y.shape: ", y.shape)
        y = F.relu(self.l2(y))

        value = self.value(y)
        adv = self.adv(y)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage

        return y


class Model_simple_linear_MLP:

    kind = "linear_mlp"  # class variable shared by all instances

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
        self.modeltype = "linear_mlp"
        self.model1 = SIMPLE_MLP(
            in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.sm = nn.Softmax(dim=1)
        self.replay = deque([], maxlen=replay_size)
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

        if epsilon > 0.2:
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().cpu().numpy())
        else:
            action = np.argmax(Q.detach().cpu().numpy())
        return action

    def training(self, exp):
        loss = torch.tensor(0)
        batch_size = 128
        if len(self.replay) > batch_size:

            minibatch = random.sample(self.replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

        #priority, (state, action, reward, next_state, done) = exp

            Q1 = self.model1(state1_batch.reshape(batch_size,18))
            Y = reward_batch.to(self.device)

            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            self.optimizer.zero_grad()
            loss = self.loss_fn(X, Y)
            loss.backward()
            self.optimizer.step()

        return loss


