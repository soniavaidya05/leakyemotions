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


class LSTM_DQN(nn.Module):
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
        n_layers=1,
        batch_first=True,
    ):
        super(LSTM_DQN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hid_size1, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.adv = nn.Linear(hid_size2, out_size)
        self.value = nn.Linear(hid_size2, 1)
        self.dropout = nn.Dropout(0.15)
        

    def forward(self, x, init_rnn_state):
        """
        TODO: check the shapes below. 
        """
        #r_in =normalize(x, p=1.0, dim = 1)
        r_in = x
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        #print("r_in.shape: ", r_in.shape)
        r_out, (h_n, h_c) = self.rnn(r_in)
        #print("r_out.shape: ", r_out.shape)
        y = F.relu(self.l1(r_out[:, -1, :]))
        # y = F.relu(self.l1(r_out))
        #print("y.shape: ", y.shape)
        y = F.relu(self.l2(y))

        value = self.value(y)
        adv = self.adv(y)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q, (h_n, h_c)



class Model_linear_LSTM_DQN:

    kind = "linear_lstm_dqn"  # class variable shared by all instances

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
        self.modeltype = "linear_lstm_dqn"
        self.model1 = LSTM_DQN(
            in_size, hid_size1, hid_size2, out_size
        )
        self.model2 = LSTM_DQN(
            in_size, hid_size1, hid_size2, out_size
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
                a=0.6,  # set this to 0 for uniform sampling (check these numbers)
                beta=0.4,  # 0.4, set this to 0 for uniform sampling (check these numbers)
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

    def pov(self, world, location, holdObject, inventory=[]):
        """
        TODO: rewrite pov to simply take in a vector
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :] = previous_state[:, 1:, :]

        #state_now = torch.tensor(inventory).float()
        #current_state[:, -1, :] = state_now

        return current_state

    def take_action(self, params):
        """
        Takes action from the input
        """

        inp, epsilon, init_rnn_state = params

        Q, (c, n) = self.model1(inp, init_rnn_state)
        p = self.sm(Q).cpu().detach().numpy()[0]

        use_softmax = False
        if use_softmax == False:
            epsilon = max(epsilon, .1)
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().cpu().numpy())

        if use_softmax == True: # AIecon simple fails with this
            if epsilon > 0.3:
                if random.random() < epsilon:
                    action = np.random.randint(0, len(p))
                else:
                    action = np.argmax(Q.detach().cpu().numpy())
            else:
                action = np.random.choice(np.arange(len(p)), p=p)

        return action, (c, n)

    def training(self, batch_size, gamma):
        """
        DQN batch learning
        """
        loss = torch.tensor(0.0)

        current_replay_size = batch_size + 1

        if current_replay_size > batch_size:

            # note, rewrite to be a min of batch_size or current_replay_size
            # need to figure out how to get current_replay_size
            minibatch, idxs, is_weight = self.PER_replay.sample(batch_size)

            # the do(device) below should not be necessary
            # but on mps, action, reward, and done are being bounced back to the cpu
            # currently removed for a test on CUDA

            state1_batch = torch.cat([s1 for (s1, a, r, s2, d, hc) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d, hc) in minibatch]).to(self.device)
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d, hc) in minibatch]).to(self.device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d, hc) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d, hc) in minibatch]).to(self.device)
            rnn_batch = torch.Tensor([d for (s1, a, r, s2, d, hc) in minibatch]).to(self.device)


            Q1, (c, n) = self.model1(state1_batch, None)
            with torch.no_grad():
                Q2, (c, n) = self.model2(state2_batch, None)

            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )

            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            errors = torch.abs(Y - X).data.cpu().numpy()

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
                    # loss = (
                    #    torch.FloatTensor(is_weight).to(self.device) * F.mse_loss(Y, X)
                    # ).mean()
                    # compute this twice!
                    loss = (
                        torch.FloatTensor(is_weight).to(self.device)
                        * ((X - Y.detach()) ** 2)
                    ).mean()
            # the step below is where the M1 chip fails
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
            exp[0],
            (
                exp[1][0].to(self.device),
                torch.tensor(exp[1][1]).float().to(self.device),
                torch.tensor(exp[1][2]).float().to(self.device),
                exp[1][3].to(self.device),
                torch.tensor(exp[1][4]).float().to(self.device),
            ),
        )

        self.PER_replay.add(exp[0], exp[1])
        if extra_reward == True and abs(high_reward) > 9:
            for _ in range(seqLength):
                self.PER_replay.add(exp[0], exp[1])
