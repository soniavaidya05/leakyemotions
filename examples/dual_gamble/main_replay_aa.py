import numpy as np
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


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

        return state, out1, out2, ev1, ev2


import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.0, capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.dqn = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity)

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state = self.buffer.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)

        next_q_values = self.dqn(next_state)
        next_q_value = torch.max(next_q_values, dim=1)[0]

        target = reward + self.gamma * next_q_value

        q_values = self.dqn(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.criterion(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim), None
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.dqn(state)
            return torch.argmax(q_values).item(), q_values


env = dual_gamble(num_gambles=2)  # change to 2 for dual gambles
state_dim = (12 * 2) + 8  # length of the concatenated one-hot vectors
action_dim = 2  # assuming 4 different actions
agent = Agent(state_dim, action_dim)

epochs = 1500000
epsilon = 1.0

state, outcome1, outcome2, ev1, ev2 = env.generate_trial()
noise1 = torch.tensor(np.random.normal(0, 0.5, 8) + 0.5).float()
noise0 = noise1 * 0
state = torch.tensor(state).float()

state_n1 = torch.cat([state, noise1])
state_n0 = torch.cat([state, noise0])

print(state)
print(state_n1)
print(state_n0)

ev1b = ev1
ev2b = ev2
done = 0
running_rewards = 0
actions = [0, 0]

Q1_diffs = []
Q2_diffs = []

for epoch in range(epochs):
    action, q_values = agent.get_action(state_n1, epsilon)
    action_announce, q_values_announce = agent.get_action(state_n0, epsilon)
    if q_values is not None and q_values_announce is not None:
        Q1 = q_values[0].item() - q_values[1].item()
        Q2 = q_values_announce[0].item() - q_values_announce[1].item()
        Q1_diffs.append(Q1)
        Q2_diffs.append(Q2)

    epsilon *= 0.99999

    if action == 0:
        reward = outcome1
    else:
        reward = outcome2
    actions[action] += 1

    running_rewards += reward
    ev1b = ev1
    ev2b = ev2
    prev_state = state_n1  # this really isn't needed, since no bellman equation is used
    state, outcome1, outcome2, ev1, ev2 = env.generate_trial()
    state = torch.tensor(state).float()
    noise1 = torch.tensor(np.random.normal(0, 0.5, 8) + 0.5).float()
    noise0 = noise1 * 0
    state_n1 = torch.cat([state, noise1])
    state_n0 = torch.cat([state, noise0])

    agent.buffer.push(prev_state.numpy(), action, reward, state_n1.numpy())

    agent.update(batch_size=64)

    if epoch % 1000 == 0:
        if q_values is not None and q_values_announce is not None:
            print(
                epoch,
                running_rewards,
                epsilon,
                actions,
                q_values.detach().numpy(),
                q_values_announce.detach().numpy(),
                ev1b,
                ev2b,
            )
            running_rewards = 0
            actions = [0, 0]

    if epoch == epochs - 1:
        done = 1  # this really isn't needed, but it's here for clarity


import pandas as pd
import matplotlib.pyplot as plt


def moving_correlation(Q1_diffs, Q2_diffs, window_size):
    # Convert the lists to pandas Series
    Q1_diffs_series = pd.Series(Q1_diffs)
    Q2_diffs_series = pd.Series(Q2_diffs)

    # Compute the moving correlation
    moving_corr = Q1_diffs_series.rolling(window=window_size).corr(Q2_diffs_series)

    # Let's save the correlations in a list, discarding the initial NaN values.
    moving_corr_values = moving_corr[~moving_corr.isna()].tolist()

    return moving_corr_values


# Use the function
window_size = 50000  # You can set the window size as required
correlations = moving_correlation(Q1_diffs, Q2_diffs, window_size)

# Plot the moving correlation
plt.figure(figsize=(14, 7))
plt.plot(correlations)
plt.ylabel("Moving Correlation")
plt.xlabel("Time")
plt.title(f"Moving Correlation with Window Size {window_size}")
plt.grid(True)
plt.show()

# Compute the standard error of correlations and 95% confidence intervals
standard_errors = [np.sqrt((1 - r**2) / (window_size - 2)) for r in correlations]
confidence_intervals = [
    (r - 1.96 * se, r + 1.96 * se) for r, se in zip(correlations, standard_errors)
]

# Plot the moving correlation along with the confidence intervals
plt.figure(figsize=(14, 7))
plt.plot(correlations, label="Moving Correlation")
plt.fill_between(
    range(len(correlations)),
    [ci[0] for ci in confidence_intervals],
    [ci[1] for ci in confidence_intervals],
    color="b",
    alpha=0.1,
    label="95% Confidence Interval",
)
plt.ylabel("Moving Correlation")
plt.xlabel("Time")
plt.title(f"Moving Correlation with Window Size {window_size}")
plt.legend()
plt.grid(True)
plt.show()
