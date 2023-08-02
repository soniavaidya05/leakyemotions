import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


class dual_gamble:
    def __init__(
        self,
        num_gambles=1,
        pos_values=[10, 7, 4, 1],
        neg_values=[-10, -7, -4, -1],
        pos_probs=[0.8, 0.6, 0.4, 0.2],
    ):
        self.pos_values = pos_values
        self.neg_values = neg_values
        self.pos_probs = pos_probs
        self.num_gambles = num_gambles

    def generate_trial(self):
        pos1 = np.random.choice(4)
        neg1 = np.random.choice(4)
        prob1 = np.random.choice(4)

        # convert pos, neg, and prob into one hot codes and concatenate them
        pos1_one_hot = np.eye(len(self.pos_values))[pos1]
        neg1_one_hot = np.eye(len(self.neg_values))[neg1]
        prob1_one_hot = np.eye(len(self.pos_probs))[prob1]

        pos1_possible = self.pos_values[pos1]
        neg1_possible = self.neg_values[neg1]
        prob1_possible = self.pos_probs[prob1]

        ev1b = prob1_possible * pos1_possible + (1 - prob1_possible) * neg1_possible

        if random.random() < prob1_possible:
            outcome1 = pos1_possible
        else:
            outcome1 = neg1_possible

        pos2_one_hot = [0, 0, 0, 0]
        neg2_one_hot = [0, 0, 0, 0]
        prob2_one_hot = [0, 0, 0, 0]
        outcome2 = 0

        if self.num_gambles == 2:
            pos2 = np.random.choice(4)
            neg2 = np.random.choice(4)
            prob2 = np.random.choice(4)

            # convert pos, neg, and prob into one hot codes and concatenate them
            pos2_one_hot = np.eye(len(self.pos_values))[pos2]
            neg2_one_hot = np.eye(len(self.neg_values))[neg2]
            prob2_one_hot = np.eye(len(self.pos_probs))[prob2]

            pos2_possible = self.pos_values[pos2]
            neg2_possible = self.neg_values[neg2]
            prob2_possible = self.pos_probs[prob2]

            if random.random() < prob2_possible:
                outcome2 = pos2_possible
            else:
                outcome2 = neg2_possible

        ev2b = prob2_possible * pos2_possible + (1 - prob2_possible) * neg2_possible

        state1 = np.hstack((pos1_one_hot, neg1_one_hot, prob1_one_hot))
        state2 = np.hstack((pos2_one_hot, neg2_one_hot, prob2_one_hot))

        out1 = outcome1
        out2 = outcome2

        counter_balance = np.random.choice(2)
        if counter_balance == 0:
            state = np.hstack((state1, state2))
            out1 = outcome1
            out2 = outcome2
            ev1 = ev1b
            ev2 = ev2b
        else:
            state = np.hstack((state2, state1))
            out1 = outcome2
            out2 = outcome1
            ev1 = ev2b
            ev2 = ev1b

        state = torch.tensor(state).float()

        return state, out1, out2


class IQN(nn.Module):
    def __init__(
        self, state_dim, action_dim, num_quantile_samples=32, embedding_dim=64
    ):
        super(IQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantile_samples = num_quantile_samples
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + embedding_dim, action_dim)

    def forward(self, state, quantiles):
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))

        batch_size = state.size(0)

        quantiles = quantiles.view(-1, 1).repeat(1, self.embedding_dim)
        i = torch.arange(1, self.embedding_dim + 1, 1).float().to(state.device)
        cos = torch.cos(quantiles * i * np.pi)
        phi = torch.relu(cos)

        phi = phi.view(batch_size, self.num_quantile_samples, self.embedding_dim)
        state = state.unsqueeze(1).expand(-1, self.num_quantile_samples, -1)

        x = torch.cat([state, phi], 2)
        x = self.fc3(x)

        return x


class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.iqn = IQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.iqn.parameters(), lr=lr)
        self.num_quantile_samples = 32
        self.num_target_quantile_samples = 32

    def update(self, state, action, reward, next_state, done):
        # ensure state and next_state are 2D (batch_size x state_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action)

        quantiles = torch.rand(self.num_quantile_samples, 1)
        next_quantiles = torch.rand(self.num_target_quantile_samples, 1)

        q_values = self.iqn(state, quantiles)
        next_q_values = self.iqn(next_state, next_quantiles).detach()  # hack for now

        target_q_values = reward + self.gamma * next_q_values * (1 - done)

        # Calculate the quantile regression loss
        diff = target_q_values - q_values
        huber_loss = torch.where(diff.abs() < 1, 0.5 * diff.pow(2), diff.abs() - 0.5)
        quantile_loss = (quantiles - (diff < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            quantiles = torch.rand(self.num_quantile_samples, 1)
            q_values = self.iqn(state, quantiles)
            return torch.argmax(q_values.mean(0)).item()


env = dual_gamble(num_gambles=2)  # change to 2 for dual gambles
state_dim = 12 * 2  # length of the concatenated one-hot vectors
action_dim = 2  # assuming 2 different actions
agent = Agent(state_dim, action_dim)

epochs = 50000000

state, outcome1, outcome2 = env.generate_trial()
done = 0
running_rewards = 0

epsilon = 0.5

for epoch in range(epochs):
    epsilon *= 0.99999

    action = agent.get_action(state, epsilon)

    if action == 0:
        reward = outcome1
    else:
        reward = outcome2

    running_rewards += reward

    prev_state = state
    state, outcome1, outcome2 = env.generate_trial()

    loss = agent.update(prev_state, action, reward, state, done)

    if epoch % 1000 == 0:
        print(epoch, loss.detach().numpy(), running_rewards / 1000, epsilon)
        running_rewards = 0
