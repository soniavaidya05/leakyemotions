data = object_memory

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Constants
sequence_length = 100
state_length = 7

# Pad data
padded_data = []
for i in range(len(data)):
    padded_sequence = [([0.0] * state_length, 0) for _ in range(sequence_length)]
    start_index = max(sequence_length - (i + 1), 0)
    for j in range(start_index, sequence_length):
        padded_sequence[j] = data[j + i + 1 - sequence_length]
    padded_data.append(padded_sequence)

# Convert to PyTorch tensors
replay_buffer = []
for sequence in padded_data:
    state_tensor = torch.tensor([state for state, _ in sequence], dtype=torch.float32)
    reward_tensor = torch.tensor(
        [reward for _, reward in sequence], dtype=torch.float32
    )
    replay_buffer.append((state_tensor, reward_tensor))


class RewardPredictor(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward, sequence_length):
        super(RewardPredictor, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(input_dim * sequence_length, sequence_length)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def sample_batch(self, replay_buffer, batch_size):
        sampled_sequences = random.sample(replay_buffer, batch_size)
        states_batch = torch.stack([states for states, _ in sampled_sequences])
        rewards_batch = torch.stack([rewards for _, rewards in sampled_sequences])
        return states_batch, rewards_batch

    def learn(self, replay_buffer, batch_size, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Sample a batch from the replay buffer
        states_batch, target_rewards_batch = self.sample_batch(
            replay_buffer, batch_size
        )

        # Forward pass
        predictions = self(states_batch)

        # Compute loss
        loss = self.loss_function(predictions, target_rewards_batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


input_dim = 7  # Dimension of the state
nhead = 1  # Number of attention heads
num_layers = 5  # Number of encoder layers
dim_feedforward = 2048  # Dimension of feedforward network
sequence_length = 100

model = RewardPredictor(input_dim, nhead, num_layers, dim_feedforward, sequence_length)

for epoch in range(1000):
    loss = model.learn(replay_buffer, 100, lr=0.001)
    print(epoch, "Training loss:", loss)
