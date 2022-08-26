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

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        r_out, (h_n, h_c) = self.rnn(r_in)

        y = F.relu(self.l1(r_out[:, -1, :]))
        y = F.relu(self.l2(y))

        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))

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

    def createInput(self, world, i, j, holdObject, numMemories=-1):
        img = agentVisualField(world, (i, j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = input.unsqueeze(0)
        return input

    def takeAction(self, params):
        inp = params
        policy, value = self.model1(inp)
        self.values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        self.logprobs.append(logprob_)
        return action

        # these need to be in the worker, not the model

        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:  # F
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)

        inp, epsilon = params
        Q, V = self.model1(inp)
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
        self.loss.backward()
        self.optimizer.step()
        return actor_loss, critic_loss, len(rewards)

        # loss = torch.tensor(0.0)
        #
        # if len(self.replay) > batch_size:
        #
        #    minibatch = random.sample(self.replay, batch_size)
        #    state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
        #    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
        #    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
        #    state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
        #    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
        #
        # Q1 = self.model1(state1_batch)
        #    with torch.no_grad():
        #        Q2 = self.model2(state2_batch)
        #
        #    Y = reward_batch + gamma * (
        #        (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
        #    )
        #    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        #
        #    self.optimizer.zero_grad()
        #    loss = self.loss_fn(X, Y.detach())
        #    loss.backward()
        #    self.optimizer.step()
        # return loss

    def updateQ(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def transferMemories(self, world, i, j, extraReward=True, seqLength=5):
        reward = 0  # needs to be added into this command
        self.rewards.append(reward)
        # transfer the events from agent memory to model replay

        # the version below should be a replication of the version in utils
        # that seems to be working. this will test if the problem is the general
        # class problem, or in the specific code commented out beelow

        # version = 3
        #
        # if version == 0:
        #    exp = world[i, j, 0].replay[-1]
        #    self.replay.append(exp)
        #    if extraReward == True and abs(exp[2]) > 9:
        #        for _ in range(5):
        #            self.replay.append(exp)
        #
        # if version == 1:
        #    # conceptually, if the selection of elements is right, this should be identical to above
        #    exp = world[i, j, 0].replay[-1]
        #    seq1 = world[i, j, 0].replay[-1][0]
        #    seq2 = world[i, j, 0].replay[-1][3]
        #    exp = (seq1, exp[1], exp[2], seq2, exp[4])
        #    self.replay.append(exp)
        #    if extraReward == True and abs(exp[2]) > 9:
        #        for _ in range(5):
        #            self.replay.append(exp)
        #
        # if version == 2:
        #    # conceptually, this should be like above, but adding one memory to state and nextstate
        #    exp = world[i, j, 0].replay[-1]
        #    state_previous = world[i, j, 0].replay[-2][0]
        #    state_now = world[i, j, 0].replay[-1][0]
        #    state_next = world[i, j, 0].replay[-1][3]
        #    seq1 = torch.cat([state_previous, state_now], dim=1)
        #    seq2 = torch.cat([state_now, state_next], dim=1)
        #    exp = (seq1, exp[1], exp[2], seq2, exp[4])
        #    self.replay.append(exp)
        #    if extraReward == True and abs(exp[2]) > 9:
        #        for _ in range(5):
        #            self.replay.append(exp)
        #
        # if version == 3:
        #    # conceptually, just like 2, but increasing the step to 4 time points
        #    exp = world[i, j, 0].replay[-1]
        #    state_previous3 = world[i, j, 0].replay[-4][0]
        #    state_previous2 = world[i, j, 0].replay[-3][0]
        #    state_previous1 = world[i, j, 0].replay[-2][0]
        #    state_now = world[i, j, 0].replay[-1][0]
        #    state_next = world[i, j, 0].replay[-1][3]
        #    seq1 = torch.cat(
        #        [state_previous3, state_previous2, state_previous1, state_now], dim=1
        #    )
        #    seq2 = torch.cat(
        #        [state_previous2, state_previous1, state_now, state_next], dim=1
        #    )
        #    exp = (seq1, exp[1], exp[2], seq2, exp[4])
        #    self.replay.append(exp)
        #    if extraReward == True and abs(exp[2]) > 9:
        #        for _ in range(5):
        #            self.replay.append(exp)
        #
        # if version == 4:
        #
        #    exp = world[i, j, 0].replay[-1]
        #
        #    state_now = world[i, j, 0].replay[-1][0]
        #    state_next = world[i, j, 0].replay[-1][3]
        #
        #    seq1 = world[i, j, 0].replay[seqLength * -1][0]
        #    seq2 = world[i, j, 0].replay[(seqLength * -1) + 1][0]
        #
        #    for mem in np.arange((seqLength * -1) + 1, -1):
        #        seq1 = torch.cat([seq1, world[i, j, 0].replay[mem][0]], dim=1)
        #        seq2 = torch.cat([seq2, world[i, j, 0].replay[mem + 1][0]], dim=1)
        #
        #    seq1 = torch.cat([seq1, state_now], dim=1)
        #    seq2 = torch.cat([seq1, state_next], dim=1)
        #
        #    exp = (seq1, exp[1], exp[2], seq2, exp[4])
        #
        #    self.replay.append(exp)
        #    if extraReward == True and abs(exp[2]) > 9:
        #        for _ in range(5):
        #            self.replay.append(exp)

        # conceptually, this is the right code below
        # exp = world[i, j, 0].replay[-1]

        # t1 = world[i, j, 0].replay[-5][3]
        # t2 = world[i, j, 0].replay[-4][3]
        # t3 = world[i, j, 0].replay[-3][3]
        # t4 = world[i, j, 0].replay[-2][3]
        # t5 = world[i, j, 0].replay[-1][3]

        # seq1 = torch.cat([t1, t2, t3, t4], dim=1)
        # seq2 = torch.cat([t2, t3, t4, t5], dim=1)

        # exp = (seq1, exp[1], exp[2], seq2, exp[4])

        # self.replay.append(exp)
        # if extraReward == True and abs(exp[2]) > 9:
        #    for _ in range(5):
        #        self.replay.append(exp)
