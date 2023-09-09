import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.optim as optim

from examples.food_trucks.models.iRainbow_clean import ReplayBuffer

class BC(nn.Module):

    def __init__(
            self,
            state_dim: ArrayLike,
            frames: int,
            act_dim: int,
            hidden_size: int,
            n_channels: int,
            batch_size: int,
            BUFFER_SIZE: int,
            GAMMA: float,
            device,
            n_step,
            seed
        ):
        
        self.state_dim = state_dim
        self.frames = frames
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.n_channels = n_channels
        self.batch_size = batch_size

        self.memory = ReplayBuffer(
            BUFFER_SIZE,
            GAMMA,
            self.batch_size,
            device,
            n_step,
            seed,
        )

        # MLP backbone
        self.layers = nn.Sequential(
            nn.Linear(
                # Pass in flattened states and actions together
                frames * np.array(state_dim).prod() + (frames - 1) * act_dim, 
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Sigmoid()
        )

        # Head for each channel of a state
        self.state_head = [
            nn.Sequential(
                nn.Linear(
                    hidden_size,
                    np.array(state_dim)[1:].prod()
                ),
                nn.Sigmoid()
            ) for _ in range(self.n_channels)
        ]

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(
                hidden_size,
                act_dim
            ),
            nn.Softmax(dim=0)
        )

    def forward(
        self, states, actions      
    ):
        # Reshape inputs
        B, T, C, H, W = states.size()
        states.view(B, -1)
        actions.view(B, -1)
        x = torch.cat(
            (states, actions),
            dim = 1
        )

        # Backbone
        x = self.layers(x)

        # Channel heads
        a1_channel = self.state_head[0](x)
        w1_channel = self.state_head[1](x)
        t1_channel = self.state_head[2](x)
        t2_channel = self.state_head[3](x)
        t3_channel = self.state_head[4](x)

        # Action head
        acts = self.action_head(x)
        
        return (
            a1_channel,
            w1_channel,
            t1_channel,
            t2_channel,
            t3_channel,
            acts
        )
    
class MLPBCModel(nn.Module):

    def __init__(self,
            state_dim: ArrayLike,
            frames: int,
            act_dim: int,
            hidden_size: int,
            n_channels: int,
            batch_size: int,
            BUFFER_SIZE: int,
            GAMMA: float,
            device,
            n_step,
            seed,
            lr
        ):

        self.model = BC(
            state_dim,
            frames,
            act_dim,
            hidden_size,
            n_channels,
            batch_size,
            BUFFER_SIZE,
            GAMMA,
            device,
            n_step,
            seed
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

    def train_model(self):

        loss = torch.tensor(0.0)
        
        if len(self.memory) > self.batch_size:

            # Both states and actions should be set up as tensors of size (B, T, -1)
            states, actions, _, _, _ = self.memory.sample()

            # Split the last action from the input frames
            action_frames, masked_action = actions[:, :-1, :], actions[:, -1:, :]

            # Split the state into channels to compute loss against
            # (Agent, Wall, Truck1, Truck2, Truck3)
            a_gt, w_gt, t1_gt, t2_gt, t3_gt = (
                states[:, 0, :, :, :], 
                states[:, 1, :, :, :], 
                states[:, 2, :, :, :],
                states[:, 3, :, :, :],
                states[:, 4, :, :, :]
            )

            # Get the model predictions
            a1, w1, t1, t2, t3, action_prediction = self.model(
                states, action_frames
            )

            agent_loss = self.loss_fn(a1, a_gt)
            wall_loss = self.loss_fn(w1, w_gt)
            t1_loss = self.loss_fn(t1, t1_gt)
            t2_loss = self.loss_fn(t2, t2_gt)
            t3_loss = self.loss_fn(t3, t3_gt)
            act_loss = self.loss_fn(action_prediction, masked_action)

            loss = agent_loss + wall_loss + t1_loss + t2_loss + t3_loss + act_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu().item()


