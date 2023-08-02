import numpy as np
import random
import torch


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

        state1 = np.hstack((pos1_one_hot, neg1_one_hot, prob1_one_hot))
        state2 = np.hstack((pos2_one_hot, neg2_one_hot, prob2_one_hot))

        state = np.hstack((state1, state2))

        # counter_balance = np.random.choice(2)
        # if counter_balance == 0:
        #    state = np.hstack((state1, state2))
        # else:
        #    state = np.hstack((state2, state1))

        state = torch.tensor(state).float()

        return state, outcome1, outcome2


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
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.dqn = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if not done:
            next_q_values = self.dqn(next_state)
            next_q_value = torch.max(next_q_values)
            target = reward + self.gamma * next_q_value
        else:
            target = reward

        q_values = self.dqn(state)
        q_value = q_values[action]

        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.dqn(state)
            return torch.argmax(q_values).item()


env = dual_gamble(num_gambles=2)  # change to 2 for dual gambles
state_dim = 12 * 2  # length of the concatenated one-hot vectors
action_dim = 2  # assuming 4 different actions
agent = Agent(state_dim, action_dim)

epochs = 500000

state, outcome1, outcome2 = env.generate_trial()
state = torch.tensor(state).float()
done = 0
running_rewards = 0

for epoch in range(epochs):
    action = agent.get_action(state)

    if action == 0:
        reward = outcome1 - outcome2
    else:
        reward = outcome2 - outcome1

    running_rewards += reward

    prev_state = state  # this really isn't needed, since no bellman equation is used
    state, outcome1, outcome2 = env.generate_trial()
    state = torch.tensor(state).float()

    loss = agent.update(prev_state, action, reward, state, done)

    if epoch % 1000 == 0:
        print(epoch, loss.detach().numpy(), running_rewards / 1000)
        running_rewards = 0

    if epoch == epochs - 1:
        done = 1  # this really isn't needed, but it's here for clarity
