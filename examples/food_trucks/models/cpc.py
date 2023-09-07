import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F

from IPython.display import clear_output
from PIL import Image
from examples.food_trucks.utils import fig2img
from gem.utils import find_instance


# Replay buffer (stored in stored memories)
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, transition):
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append(transition)
        self.priorities.append(max_priority)

    def sample(self, batch_size, device):
        self.frame += 1
        beta = min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

        priorities = np.array(self.priorities)
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

cnn_config = [
        {
            "layer_type": "conv",
            "in_channels": 7,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 3,
            "padding": 0,
        },
        {"layer_type": "relu"},
        {
            "layer_type": "conv",
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
        },
    ]

class Encoder(nn.Module):
    '''
    Encoder architecture
    '''

    def __init__(
        self,
        encoder_layers,
        linear_layers,
        action_layers,
        linear
    ):
        
        super().__init__()
        self.encoder_layers = encoder_layers
        self.linear_layers = linear_layers
        self.action_layers = action_layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.linear = linear

    def forward(self, x):

        # If using linear instead of cnn
        if self.linear:
            batch_size = x.size()[0]
            x = x.view(batch_size, -1)

            # Through linear layers
            x = self.encoder_layers(x)
            x = self.relu(x)
            x = self.linear_layers(x)

        else:
            # Through the encoder
            x = self.encoder_layers(x)
            x = self.relu(x)
            # Reshape the input to go through the linear layers
            batch_size, _, _, _ = x.size()
            x = x.view(batch_size, -1)
            x = self.linear_layers(x)

        # Then, create the output heads: state, state+1, state+2...
        z = self.sigmoid(x)
        zp = self.sigmoid(x)
        zpp = self.sigmoid(x)

        # ... and action predictions
        actions = self.action_layers(x)
        actions = self.softmax(actions)

        return z, zp, zpp, actions

class Decoder(nn.Module):
    '''
    Decoder architecture
    '''
    def __init__(
        self,
        linear_layers,
        decoder_layers,
        shape
    ):
        super().__init__()
        self.linear_layers = linear_layers
        self.decoder_layers = decoder_layers
        self.channels = shape[0]
        self.height = shape[1]
        self.width = shape[2]
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear_layers(x)
        x = self.relu(x)

        # Reshape the input to go through the decoder layers
        batch_size, _ = x.size()
        x = x.view(batch_size, self.channels, self.height, self.width)
        x = self.decoder_layers(x)

        # Sigmoid transformation
        x = self.sigmoid(x)

        return x

# World model
class WorldModel(nn.Module):
    '''
    Contrastive predictive coding model.

    Parameters
    ----------
    input_shape: A 3-dimensional array indicating the size of the CNN input.
    cpc_output: An integer indicating the output size of the CPC head
    action_space: An integer indicating the number of actions
    batch_size: An integer indicating the number of memories to batch
    cnn_config: A configuration dictionary describing the encoder-decoder architecture.
    stored_memories: A PrioritizedReplayMemory object with expert memories.
    '''

    def __init__(
        self,
        input_shape,
        cpc_output,
        action_space,
        batch_size,
        cnn_config,
        memories,
        memory_size,
        device,
        no_cnn = False
    ):
        
        super(WorldModel, self).__init__()

        self.input_shape = np.array(input_shape)
        self.cpc_output = cpc_output
        self.action_space = action_space
        self.batch_size = batch_size
        self.cnn_config = cnn_config
        self.memories = memories
        self.memory_size = memory_size
        self.loss_fn = nn.MSELoss()
        self.device = device

        # Build encoder and optimizer for encoder architecture
        self.encoder = self.build_encoder(linear=no_cnn)
        self.optim_e = optim.Adam(self.encoder.parameters(), lr = .001)
        

        # NOTE: the decoder uses the detached z from the encoder
        self.decoder = self.build_decoder()
        self.optim_d = optim.Adam(self.decoder.parameters(), lr = .001)

    def build_linear_layers(self, reversed = False, linear = False):

        if linear:
            in_size = self.cpc_output * 8
        else:
            in_size = self.cnn_config[0]['out_channels'] * self.cnn_config[0]['kernel_size'] * self.cnn_config[0]['stride']

        layer_config = [
            {
                'layer_type': 'linear',
                'in_features': in_size,
                'out_features': self.cpc_output * 8
            },
            {'layer_type': 'relu'},
            {
                'layer_type': 'linear',
                'in_features': self.cpc_output * 8,
                'out_features': self.cpc_output * 4
            },
            {'layer_type': 'sigmoid'},
            {
                'layer_type': 'linear',
                'in_features': self.cpc_output * 4,
                'out_features': self.cpc_output
            },
        ]

        layers = []
    
        # Swap input and output layer sizes
        if reversed:
            inp = 'out_features'
            out = 'in_features'
            layer_config = layer_config[::-1]
        else:
            inp = 'in_features'
            out = 'out_features'

        # Build layers
        for config in layer_config:
            if config['layer_type'] == 'linear':
                layer = nn.Linear(
                    in_features = config[inp],
                    out_features = config[out]
                )
            elif config['layer_type'] == 'relu':
                layer = nn.ReLU()
            elif config['layer_type'] == 'sigmoid':
                layer = nn.Sigmoid()
                
            else:
                raise ValueError(f"Invalid layer_type: {config['layer_type']}")
            layers.append(layer)

        return nn.Sequential(*layers)

    def build_encoder(self, linear = False):
        '''
        Build the encoder network using the CNN configuration.
        '''


        # First, build the encoder layers
        layers = []

        if not linear:
        # Add layers through the cnn config
            for config in self.cnn_config:
                if config['layer_type'] == 'conv':
                    layer = nn.Conv2d(
                        in_channels = config['in_channels'],
                        out_channels = config['out_channels'],
                        kernel_size = config['kernel_size'],
                        stride = config['stride'],
                        padding = config['padding']
                    )
                elif config['layer_type'] == 'relu':
                    layer = nn.ReLU()
                elif config["layer_type"] == "pool":  # Add this block
                    layer = nn.MaxPool2d(
                        kernel_size=config["kernel_size"], stride=config["stride"]
                    )
                elif config["layer_type"] == "dropout":  # Add this block
                    layer = nn.Dropout2d(p=config["dropout_rate"])
                else:
                    raise ValueError(f"Invalid layer_type: {config['layer_type']}")
                layers.append(layer)

            encoder_layers = nn.Sequential(*layers)
        # Alternate: Build linear layers
        else:
            layers.append(
                nn.Linear(
                    in_features=self.memory_size * self.input_shape.prod(),
                    out_features=self.memory_size * self.input_shape.prod()
                )
            )
            layers.append(
                nn.ReLU(),
            )
            layers.append(
                nn.Linear(
                    in_features=self.memory_size * self.input_shape.prod(),
                    out_features=self.cpc_output * 8
                )
            )
            encoder_layers = nn.Sequential(*layers)

        # Next, build the linear layers
        linear_layers = self.build_linear_layers(linear=linear)

        action_layers = nn.Sequential(
            nn.Linear(
                in_features = self.cpc_output,
                out_features = self.action_space
            )
        )

        net = Encoder(
            encoder_layers,
            linear_layers,
            action_layers,
            linear=linear
        )

        return net
    
    def build_decoder(self):
        '''
        Build the decoder network using the CNN configuration
        '''
        
        # First, create the linear layers
        linear_layers = self.build_linear_layers(reversed=True)
        
        layers = []
        reversed_cnn_config = self.cnn_config[::-1]

        for config in reversed_cnn_config:
            if config["layer_type"] == "conv":
                layer = nn.ConvTranspose2d(
                    in_channels=config['out_channels'],
                    out_channels=config["in_channels"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config.get("padding", 0),  # Add this line
                )
            elif config["layer_type"] == "relu":
                layer = nn.ReLU()
            elif config["layer_type"] == "pool":
                layer = nn.Upsample(
                    scale_factor=config["stride"], mode="bilinear", align_corners=True
                )  # Modify this line
            elif config["layer_type"] == "dropout":
                layer = nn.Dropout2d(p=config["dropout_rate"])
            else:
                raise ValueError(f"Invalid layer_type: {config['layer_type']}")
            layers.append(layer)

        decoder_layers = nn.Sequential(*layers)

        shape = (cnn_config[0]['out_channels'], cnn_config[0]['kernel_size'], cnn_config[0]['stride'])

        net = Decoder(
            linear_layers,
            decoder_layers,
            shape
        )

        return net

    def compute_cpc_loss(
        self,
        latent_state,
        latent_next_state
    ):
        '''
        Compute contrastive predictive coding loss between ẑ' and z'
        '''

        # Dot product between current and future latent states (positive sample)
        pos_dot_product = (latent_state * latent_next_state).sum(dim=-1)

        # Dot product between orthogonal states (negative sample)
        neg_dot_product = (
            (latent_state.unsqueeze(1) * latent_next_state.unsqueeze(0))
            .sum(dim=-1)
            .mean(dim=0)
        )

        cpc_loss = -F.log_softmax(pos_dot_product - neg_dot_product, dim = -1).mean(dim = 0)

        return cpc_loss
    
    def train(self):
        if len(self.memories) < self.batch_size:
            return 0, 0, 0
        
        samples, indices, weights = self.memories.sample(self.batch_size, self.device)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states).to(self.device).squeeze() / 255
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(
            -1
        )
        next_states = torch.stack(next_states).to(self.device).squeeze() / 255
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1)
        weights = weights.unsqueeze(-1)


        # Encoder of current states
        z, zp_hat, zpp_hat, predicted_actions = self.encoder(states)

        # Get future embedding from next states
        zp, zpp, _, _ = self.encoder(next_states)

        # Compute cpc loss
        cpc_loss = self.compute_cpc_loss(zp_hat, zp)


        latent_states = z.detach()
        latent_next_states = zp.detach()

        predicted_states = self.decoder(latent_states)
        predicted_next_states = self.decoder(latent_next_states)

        r1_loss = self.loss_fn(
            predicted_states,
            states[:, self.memory_size - 1, :, :, :]
        )

        r2_loss = self.loss_fn(
            predicted_next_states,
            next_states[:, self.memory_size - 1, :, :, :]
        )

        # Backpropagate loss
        loss = r1_loss + r2_loss

        cpc_loss.backward()
        loss.backward()


        # Zero gradients
        self.optim_e.zero_grad()
        self.optim_d.zero_grad()

        self.memories.update_priorities(
            indices, loss.detach().cpu().numpy().reshape(-1)
        )
        self.optim_e.step()
        self.optim_d.step()

        return cpc_loss.item(), r1_loss.item(), r2_loss.item()

    def update_memory(self, state, action, reward, next_state, done):
        self.memories.push((state, action, reward, next_state, done))

def plot_states(actual, predicted, time, one_hot = False, filepath = '/Users/rgelpi/Documents/GitHub/transformers/examples/food_trucks/data/states_0.png'):
    if not one_hot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(actual.astype(int).transpose(1, 2, 0))
        axes[0].set_title(f"Actual {time[0:3]} state")
        axes[1].imshow(predicted.astype(int).transpose(1, 2, 0))
        axes[1].set_title(f"Predicted {time[0:3]} state")
        plt.show()
        img = fig2img(fig)
        return img
    else:

        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes[0,0].imshow(actual.astype(int)[2, :, :])
        axes[0,0].set_title(f'A. {time[0:3]}: Agent')
        axes[0,1].imshow(actual.astype(int)[1, :, :])
        axes[0,1].set_title(f'A. {time[0:3]}: Wall')
        axes[0,2].imshow(actual.astype(int)[3, :, :])
        axes[0,2].set_title(f'A. {time[0:3]}: Korean')
        axes[0,3].imshow(actual.astype(int)[4, :, :])
        axes[0,3].set_title(f'A. {time[0:3]}: Lebanese')        
        axes[0,4].imshow(actual.astype(int)[5, :, :])
        axes[0,4].set_title(f'A. {time[0:3]}: Mexican')
        axes[1,0].imshow(predicted.astype(int)[2, :, :], vmin = 0., vmax = 1.)
        axes[1,0].set_title(f'P. {time[0:3]}: Agent')
        axes[1,1].imshow(predicted.astype(int)[1, :, :], vmin = 0., vmax = 1.)
        axes[1,1].set_title(f'P. {time[0:3]}: Wall')
        axes[1,2].imshow(predicted.astype(int)[3, :, :], vmin = 0., vmax = 1.)
        axes[1,2].set_title(f'P. {time[0:3]}: Korean')
        axes[1,3].imshow(predicted.astype(int)[4, :, :], vmin = 0., vmax = 1.)
        axes[1,3].set_title(f'P. {time[0:3]}: Lebanese')        
        axes[1,4].imshow(predicted.astype(int)[5, :, :], vmin = 0., vmax = 1.)
        axes[1,4].set_title(f'P. {time[0:3]}: Mexican')
        plt.show()
        img = fig2img(fig)
        return img

def plot_q_values(q_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(q_values)), q_values)
    ax.set_xticks(range(len(q_values)))
    ax.set_title("Q-values")
    plt.show()

def sample_and_visualize_test(model, 
                              device, 
                              one_hot = False, 
                              epoch = 0, 
                              filepath = '/Users/rgelpi/Documents/GitHub/transformers/examples/food_trucks/data/states',
                              action_model = None,
                              env = None):
    
    filepath = filepath + '_' + str(epoch) + '.png'
    video_path = filepath + '_' + str(epoch) + '.gif'
    video_path2 = filepath + '_' + str(epoch) + '_2' + '.gif'

    imgs = []
    imgs2 = []
    
    if action_model == None:
        sample, _, _ = model.memory.sample(30, device)

        fig = plt.figure()
        fig2 = plt.figure()

        for each_sample in sample:
            state, action, _, next_state, _ = each_sample
            state_tensor = state.unsqueeze(0).to(device).squeeze().unsqueeze(0)
            # Remove empty layer

            with torch.no_grad():
                latent, _, _, _ = model.encoder(state_tensor)
                latent2, _, _, _ = model.encoder(state_tensor)
                predicted_state = model.decoder(latent.detach())
                predicted_next_state = model.decoder(latent2.detach())
                

            predicted_next_state = predicted_next_state.cpu().numpy()
            actual_next_state = next_state.cpu().squeeze().numpy()

            predicted_state = predicted_state.cpu().numpy()
            actual_state = state.cpu().squeeze().numpy()

            # remove empty layer
            actual_state = actual_state[model.memory_size - 1, :, :, :]
            actual_next_state = actual_next_state[model.memory_size - 1, :, :, :]
            predicted_state = predicted_state
            predicted_next_state = predicted_next_state

            img = plot_states(actual_state, predicted_state, 'current', one_hot, filepath)
            img2 = plot_states(actual_next_state, predicted_next_state, 'next', one_hot, filepath)

            imgs.append(img)
            imgs2.append(img2)

    else:

        env.reset_env()
        done = 0
        turn = 0

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            agent = env.world[loc] # Agent at this location
            agent.init_replay(
                5,
                one_hot = env.one_hot
            )
            agent.init_rnn_state = None

        while done == 0:

            turn += 1

            agentList = find_instance(env.world, "neural_network")
            random.shuffle(agentList)

            # For each agent, act and get the transitions and experience
            for loc in agentList:
                agent = env.world[loc]
                agent.reward = 0
                state = env.pov(loc)

                action = action_model[agent.policy].take_action(state, 0)

                (
                    env.world,
                    _,
                    next_state,
                    done,
                    _,
                ) = agent.transition(env, action_model, action[0], loc)

                state_tensor = state.unsqueeze(0).to(device).squeeze().unsqueeze(0)
                    
                with torch.no_grad():
                    latent, _, _, _ = model.encoder(state_tensor)
                    latent2, _, _, _ = model.encoder(state_tensor)
                    predicted_state = model.decoder(latent.detach())
                    predicted_next_state = model.decoder(latent2.detach())

                predicted_next_state = predicted_next_state.squeeze().cpu().numpy()
                actual_next_state = next_state.cpu().squeeze().numpy()

                predicted_state = predicted_state.squeeze().cpu().numpy()
                actual_state = state.cpu().squeeze().numpy()



                # remove empty layer
                actual_state = actual_state[model.memory_size - 1, :, :, :]
                actual_next_state = actual_next_state[model.memory_size - 1, :, :, :]
                

                img = plot_states(actual_state, predicted_state, 'current', one_hot, filepath)
                img2 = plot_states(actual_next_state, predicted_next_state, 'next', one_hot, filepath)

                imgs.append(img)
                imgs2.append(img2)

                if turn >= 100:
                    done = 1
                

    imgs[0].save(video_path, format = 'GIF', append_images = imgs[1:], save_all = True, duration = 100, loop = 0)
    imgs2[0].save(video_path2, format = 'GIF', append_images = imgs2[1:], save_all = True, duration = 100, loop = 0)

    




