# --------------- #
# region: Imports #
# --------------- #

import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from examples.ft.models.iqn import ReplayBuffer
from examples.ft.models.ann import ANN

# --------------- #
# endregion       #
# --------------- #

class BC(nn.Module):
    '''
    Behavioural cloning network.
    '''

    def __init__(
            self,
            state_size: ArrayLike,
            action_size: int,
            layer_size: int,
            n_channels: int,
            frames: int
        ):

        super(BC, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.n_channels = n_channels
        self.frames = frames

        # MLP backbone
        self.layers = nn.Sequential(
            nn.Linear(
                # Pass in flattened states and actions together
                frames * np.array(state_size).prod() + (frames - 1) * action_size, 
                layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                layer_size,
                layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                layer_size,
                layer_size
            ),
            nn.Sigmoid()
        )

        # Head for each channel of a state
        self.state_head = [
            nn.Sequential(
                nn.Linear(
                    layer_size,
                    np.array(state_size)[1:].prod()
                ),
                nn.Sigmoid()
            ) for _ in range(self.n_channels)
        ]

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(
                layer_size,
                action_size
            ),
            nn.Softmax(dim=0)
        )

    def forward(
        self, states, actions      
    ):
        # Reshape inputs
        B, T, C, H, W = states.size()
        states = states.view(B, -1)
        actions = actions.view(B, -1)
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
    
class MLPBCModel(ANN):
    '''
    MLP-based behavioural cloning model with experience replay.
    '''

    def __init__(
        self,
        # Base ANN parameters
        state_size: ArrayLike,
        action_size: int,
        layer_size: int,
        epsilon: float,
        device: Union[str, torch.device],
        seed: int,
        # BC model parameters
        n_channels: int,
        frames: int,
        memory: ReplayBuffer,
        LR: float
        ):

        super(MLPBCModel, self).__init__(state_size, action_size, layer_size, epsilon, device, seed)
        
        self.model = BC(
            state_size,
            action_size,
            layer_size,
            n_channels,
            frames
        )

        self.memory = memory
        self.frames = frames
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = LR)

    def train_model(self):

        loss = torch.tensor(0.0)
        
        if len(self.memory) > self.memory.batch_size:

            # Both states and actions should be set up as tensors of size (B, T, -1)
            states, actions, _, _, _ = self.memory.sample()

            # Split the last action from the input frames
            action_frames, masked_action = actions[:, :-1, :], actions[:, -1, :]

            # Remove the extra dimension from states
            states = states.squeeze()
            B, T, C, H, W = states.size()

            # Split the state into channels to compute loss against
            # (Agent, Wall, Truck1, Truck2, Truck3)
            a_gt, w_gt, t1_gt, t2_gt, t3_gt = (
                states[:, self.frames - 1, 0, :, :].reshape(B, -1), 
                states[:, self.frames - 1, 1, :, :].reshape(B, -1), 
                states[:, self.frames - 1, 2, :, :].reshape(B, -1),
                states[:, self.frames - 1, 3, :, :].reshape(B, -1),
                states[:, self.frames - 1, 4, :, :].reshape(B, -1)
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
            act_loss = self.loss_fn(action_prediction, masked_action.to(float))

            loss = agent_loss + wall_loss + t1_loss + t2_loss + t3_loss + act_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu().item()


