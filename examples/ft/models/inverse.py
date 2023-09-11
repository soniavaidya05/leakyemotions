# --------------- #
# region: Imports #
# --------------- #

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from numpy.typing import ArrayLike
from typing import Union
from matplotlib import pyplot as plt

from examples.ft.models.buffer import ReplayBuffer
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
        # self.state_head = [
        #     nn.Sequential(
        #         nn.Linear(
        #             layer_size,
        #             np.array(state_size)[1:].prod()
        #         ),
        #         nn.Sigmoid()
        #     ) for _ in range(self.n_channels)
        # ]

        self.state_head = nn.Sequential(
            nn.Linear(layer_size, np.array(state_size).prod()),
            nn.Sigmoid()
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(
                layer_size,
                action_size
            ),
            nn.Softmax(dim=1)
        )

    def forward(
        self, states, actions      
    ) -> tuple[torch.Tensor]:
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
        # a1_channel = self.state_head[0](x)
        # w1_channel = self.state_head[1](x)
        # t1_channel = self.state_head[2](x)
        # t2_channel = self.state_head[3](x)
        # t3_channel = self.state_head[4](x)
        state = self.state_head(x)

        # Action head
        acts = self.action_head(x)
        
        return (
            state,
            # a1_channel,
            # w1_channel,
            # t1_channel,
            # t2_channel,
            # t3_channel,
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
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = LR)

    def generate_images(self, state: torch.Tensor, action: torch.Tensor) -> list:
        '''
        Generate model outputs for a given state.

        Parameters:
            state: An individual state of dimensions T x C x H x W. \n
            action: An individual action batch of dimensions T x A.

        Returns:
            A model prediction tensor of C x H x W.
        '''
        B, T, C, H, W = state.size()

        action_input, masked_action = action[:-1, :], action[-1, :]

        with torch.no_grad():
            state_prediction, action_prediction = self.model(
                state, action_input
            )

        state = state.squeeze()[T-1, :, :, :]
        state_prediction = state_prediction.view(*state.size())

        return state_prediction, state, action_prediction, masked_action

    def plot_images(self, state: torch.Tensor, action: torch.Tensor) -> None:
        '''
        Plots model outputs for a given state.

        Parameters:
            state: An individual state of dimensions T x C x H x W. \n
            action: An individual action batch of dimensions T x A.
        '''

        state_predictions, states, action_prediction, masked_action = self.generate_images(state, action)

        states = (state_predictions, states)
        masked_action = masked_action.numpy().astype(int).tolist().index(1)
        action_prediction = np.round(action_prediction.numpy().tolist()[0], 2)

        plot_dims = (2, 5)
        fig, axes = plt.subplots(*plot_dims, figsize=(10,5))
        
        outcome = ['Pred. ', 'Act. ']
        layer = ['Agent', 'Wall', 'Korean', 'Lebanese', 'Mexican']

        fig.suptitle(f'Action: {masked_action}. Prediction: {action_prediction}')
        for x in range(plot_dims[0]):
            for y in range(plot_dims[1]):
                axes[x, y].imshow(states[x][y, :, :])
                axes[x, y].set_title(outcome[x] + layer[y])
        return fig

    def train_model(self) -> tuple[torch.Tensor]:

        loss = torch.tensor(0.0)
        
        if len(self.memory) > self.memory.batch_size:

            # Both states and actions should be set up as tensors of size (B, T, -1)
            states, actions, _, _, _ = self.memory.sample()

            # Split the last action from the input frames
            action_frames, masked_action = actions[:, :-1, :], actions[:, -1, :]

            # Remove the extra dimension from states
            states = states.squeeze()
            B, T, C, H, W = states.size()

            ground_truth = states[:, self.frames - 1, :, :, :].reshape(B, -1).to(self.device)

            state_prediction, action_prediction = self.model(
                states, action_frames
            )

            act_loss = self.loss_fn(action_prediction.to(float), masked_action.to(float))
            reconstruction_loss = self.loss_fn(state_prediction, ground_truth)


            # # Split the state into channels to compute loss against
            # # (Agent, Wall, Truck1, Truck2, Truck3)
            # a_gt, w_gt, t1_gt, t2_gt, t3_gt = (
            #     states[:, self.frames - 1, 0, :, :].reshape(B, -1).to(self.device), 
            #     states[:, self.frames - 1, 1, :, :].reshape(B, -1).to(self.device), 
            #     states[:, self.frames - 1, 2, :, :].reshape(B, -1).to(self.device),
            #     states[:, self.frames - 1, 3, :, :].reshape(B, -1).to(self.device),
            #     states[:, self.frames - 1, 4, :, :].reshape(B, -1).to(self.device)
            # )

            # # Get the model predictions
            # a1, w1, t1, t2, t3, action_prediction = self.model(
            #     states, action_frames
            # )

            # agent_loss = self.loss_fn(a1, a_gt)
            # wall_loss = self.loss_fn(w1, w_gt)
            # t1_loss = self.loss_fn(t1, t1_gt)
            # t2_loss = self.loss_fn(t2, t2_gt)
            # t3_loss = self.loss_fn(t3, t3_gt)
            # act_loss = self.loss_fn(action_prediction.to(float), masked_action.to(float))

            # reconstruction_loss = agent_loss + wall_loss + t1_loss + t2_loss + t3_loss
            loss = reconstruction_loss + act_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return act_loss.detach().cpu().item(), reconstruction_loss.detach().cpu().item()

    


