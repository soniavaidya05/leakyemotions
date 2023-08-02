import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DualGamble:
    def __init__(
        self,
        pos_values=[10, 7, 4, 1],
        neg_values=[-10, -7, -4, -1],
        pos_probs=[0.8, 0.6, 0.4, 0.2],
    ):
        self.pos_values = pos_values
        self.neg_values = neg_values
        self.pos_probs = pos_probs

    def generate_trial(self):
        pos = np.random.choice(len(self.pos_values))
        neg = np.random.choice(len(self.neg_values))
        prob = np.random.choice(len(self.pos_probs))

        pos_one_hot = np.eye(len(self.pos_values))[pos]
        neg_one_hot = np.eye(len(self.neg_values))[neg]
        prob_one_hot = np.eye(len(self.pos_probs))[prob]
        state = np.hstack((pos_one_hot, neg_one_hot, prob_one_hot))

        prob_possible = self.pos_probs[prob]

        if random.random() < prob_possible:
            outcome = self.pos_values[pos]
        else:
            outcome = self.neg_values[neg]

        return state, outcome


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
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.iqn = IQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.iqn.parameters(), lr=lr)
        self.num_quantile_samples = 32
        self.num_target_quantile_samples = 32

    def update(self, states, actions, rewards, next_states, dones):
        # ensure states and next_states are 2D (batch_size x state_dim)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions)

        batch_size = states.shape[0]

        quantiles = torch.rand(batch_size, self.num_quantile_samples, 1)
        next_quantiles = torch.rand(batch_size, self.num_target_quantile_samples, 1)

        q_values = self.iqn(states, quantiles)
        next_q_values = self.iqn(next_states, next_quantiles).detach()

        # Reshape rewards and dones tensors to be broadcastable with next_q_values
        rewards = rewards.view(-1, 1, 1)
        dones = dones.view(-1, 1, 1)

        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

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


env = DualGamble()
replay_buffer = ReplayBuffer()
state_dim = 12  # length of the concatenated one-hot vectors
action_dim = 2  # assuming 2 different actions
agent = Agent(state_dim, action_dim)
batch_size = 64  # Batch size for the replay buffer

epochs = 50000

state, outcome = env.generate_trial()
done = 0
running_rewards = 0

for epoch in range(epochs):
    action = agent.get_action(state)

    if action == 0:
        reward = outcome
    else:
        reward = -outcome

    running_rewards += reward

    prev_state = state
    state, outcome = env.generate_trial()
    done = 0

    # Push the experience to the replay buffer
    replay_buffer.push(prev_state, action, reward, state, done)

    # If the buffer size is greater than batch_size, sample experiences and update the model
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        loss = agent.update(states, actions, rewards, next_states, dones)
        if epoch % 1000 == 0:
            print(epoch, loss.detach().numpy(), running_rewards)
            running_rewards = 0
