from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from models.memory import Memory
from models.perception import agent_visualfield


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
        self.replay = deque([], maxlen=replay_size)
        self.sm = nn.Softmax(dim=1)

    def pov(self, world, location, holdObject, inventory=[], layers=[0]):
        """
        Creates outputs of a single frame, and also a multiple image sequence
        TODO: get rid of the holdObject input throughout the code
        """

        previous_state = holdObject.replay[-1][0]
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

    def training(self, batch_size, gamma, priority_replay=True):
        """
        DQN batch learning
        """

        loss = torch.tensor(0.0)

        # note, there may be a ratio of priority replay to random replay that could be ideal

        if len(self.replay) > batch_size:
            if priority_replay == False:
                minibatch = random.sample(self.replay, batch_size)
            if priority_replay == True:
                losses = self.surprise(self.replay, gamma)
                sample_indices, importance_normalized = self.priority_sample(
                    losses, sample_size=256, alpha_scaling=0.7, offset=0.1
                )
                minibatch = [self.replay[i] for i in sample_indices]

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
            if priority_replay == False:
                loss = self.loss_fn(X, Y.detach())
            if priority_replay == True:
                loss = (X - Y.detach()) ** 2 * torch.Tensor(importance_normalized)
                loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()
        return loss

    def surprise(self, replay, gamma):
        """
        DQN priority surprise
        """

        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in replay])
        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in replay])
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in replay])
        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in replay])
        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in replay])

        Q1 = self.model1(state1_batch)
        with torch.no_grad():
            Q2 = self.model2(state2_batch)

        Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2.detach(), dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        # or, just compute here for every element?

        losses = (X.detach() - Y.detach()) ** 2

        # should the normalization be done here? or later in the processing

        return losses

    def priority_sample(
        self, replay_loss, sample_size=150, alpha_scaling=0.7, offset=0.1
    ):
        """
        normalize the DQN priority surprise
        """
        replay_loss = np.asarray(replay_loss) + offset
        sample_probs = abs(replay_loss**alpha_scaling) / np.sum(
            abs(replay_loss**alpha_scaling)
        )

        importance = 1 / len(replay_loss) * 1 / sample_probs
        importance_normalized = importance / max(importance)

        sample_indices = random.choices(
            range(len(replay_loss)), k=sample_size, weights=sample_probs
        )

        importance = 1 / len(replay_loss) * 1 / sample_probs[sample_indices]
        importance_normalized = importance / max(importance)

        # the importance normalized is needed in the training function
        # I'm not sure what it does yet, so it is just here

        return sample_indices, importance_normalized

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
        exp = world[loc].replay[-1]
        self.replay.append(exp)
        if extra_reward == True and abs(exp[2]) > 9:
            for _ in range(5):
                self.replay.append(exp)
