import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

from IPython.display import clear_output
from PIL import Image
from examples.food_trucks.utils import fig2img
from gem.utils import find_instance



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

class WorldModel(nn.Module):
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
                self.beta_start
                + self.frame * (1.0 - self.beta_start) / self.beta_frames,
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

    def __init__(
        self,
        capacity,
        input_shape,
        output_shape,
        num_actions,
        batch_size,
        cnn_config,
        stored_memories,
    ):
        super(WorldModel, self).__init__()

        self.input_channels = input_shape[
            0
        ]  # Assuming input_shape is in the format (channels, height, width)

        self.capacity = capacity

        # self.memory = self.PrioritizedReplayMemory(capacity)
        self.memory = stored_memories
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.output_shape = output_shape
        self.cnn_config = cnn_config
        self.output_channels = cnn_config[-1]["out_channels"]
        self.features = self.build_encoder_layers(cnn_config)
        self.decoder = self.build_decoder_layers(cnn_config)
        self.batch_size = batch_size
        self.online_net = self.build_network(cnn_config)
        lr = 0.001
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def build_encoder_layers(self, cnn_config):
        layers = []
        prev_channels = self.input_channels

        for config in cnn_config:
            if config["layer_type"] == "conv":
                layer = nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=config["out_channels"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config.get("padding", 0),  # Add this line
                )
                prev_channels = config["out_channels"]
            elif config["layer_type"] == "relu":
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

        return nn.Sequential(*layers)

    def build_decoder_layers(self, cnn_config):
        layers = []
        reversed_cnn_config = cnn_config[::-1]
        prev_channels = self.output_channels

        for config in reversed_cnn_config:
            if config["layer_type"] == "conv":
                layer = nn.ConvTranspose2d(
                    in_channels=prev_channels,
                    out_channels=config["in_channels"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config.get("padding", 0),  # Add this line
                    output_padding=config.get("output_padding", 0),  # Add this line
                )
                prev_channels = config["in_channels"]
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

        return nn.Sequential(*layers)

    def build_linear_layers(self, input_dim, linear_config):
        layers = []
        in_features = input_dim

        for layer_config in linear_config:
            layers.append(nn.Linear(in_features, layer_config["out_features"]))
            layers.append(nn.ReLU())
            in_features = layer_config["out_features"]

        return nn.Sequential(*layers)

    def build_encoder_decoder(self):
        class CustomNetwork(nn.Module):
            def __init__(self, features, decoder):
                super().__init__()
                self.features = nn.Sequential(*features)
                self.decoder = nn.Sequential(*decoder)

            def forward(self, x):
                features = self.features(x)
                features = F.relu(features)
                decoded = self.decoder(features)

                return decoded

        features = self.features
        decoder = self.decoder

        net = CustomNetwork(features, decoder)
        return net

    def build_network(self, cnn_config):
        class CustomNetwork(nn.Module):
            def __init__(self, features, decoder):
                super().__init__()
                self.features = nn.Sequential(*features)
                self.decoder = nn.Sequential(*decoder)
                self.decoder2 = nn.Sequential(*decoder)

                self.layer_ae1 = nn.Linear(
                    in_features=cnn_config[0]['out_channels'] * cnn_config[0]['kernel_size'] * cnn_config[0]['stride'], 
                    out_features=512
                )
                self.layer_ae2 = nn.Linear(
                    in_features=512, 
                    out_features=cnn_config[0]['out_channels'] * cnn_config[0]['kernel_size'] * cnn_config[0]['stride']
                )

                self.layer_dc1 = nn.Linear(
                    in_features=512, 
                    out_features=512
                )
                self.layer_dc2 = nn.Linear(
                    in_features=512, 
                    out_features=cnn_config[0]['out_channels'] * cnn_config[0]['kernel_size'] * cnn_config[0]['stride']
                )

            def forward(self, x):
                features = self.features(x)
                features = F.relu(features)
                batch_size, channels, height, width = features.size()
                
                ae1 = F.relu(self.layer_ae1(features.view(batch_size, -1)))
                ae2 = F.relu(self.layer_ae2(ae1))
                decoded = self.decoder(ae2.view(batch_size, channels, height, width))

                # decoded = self.decoder(features)

                dc1 = F.relu(self.layer_dc1(ae1))
                dc2 = F.relu(self.layer_dc2(dc1))

                # features = features.view(x.size(0), -1)  # torch.Size([1, 1296])
                # x = F.relu(self.layer1(features))

                # x2 = F.relu(self.layer2(x))
                x2 = dc2.reshape([batch_size, channels, height, width])
                decoded2 = self.decoder2(x2)

                decoded = F.sigmoid(decoded)
                decoded2 = F.sigmoid(decoded2)

                return decoded, decoded2

        features = self.features
        decoder = self.decoder

        net = CustomNetwork(features, decoder)
        return net
    
    def build_action_network(self, cnn_config):
        class CustomNetwork(nn.Module):
            def __init__(self, features, decoder):
                super().__init__()
                self.features = nn.Sequential(*features)
                self.decoder2 = nn.Sequential(*decoder)

                self.layer_ae1 = nn.Linear(
                    in_features=cnn_config[0]['out_channels'] * cnn_config[0]['kernel_size'] * cnn_config[0]['stride'], 
                    out_features=512
                )
                self.layer_ae2 = nn.Linear(
                    in_features=512, 
                    out_features=512
                )

                self.layer_action = nn.Linear(
                    in_features=512, 
                    out_features=4
                )

            def forward(self, x):
                features = self.features(x)
                features = F.relu(features)
                batch_size, channels, height, width = features.size()
                
                ae1 = F.relu(self.layer_ae1(features.view(batch_size, -1)))
                ae2 = F.relu(self.layer_ae2(ae1))
                action = F.relu(self.layer_action(ae2))

                return action

        features = self.features
        decoder = self.decoder

        net = CustomNetwork(features, decoder)
        return net

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def train_model(self, device):
        if len(self.memory) < self.batch_size:
            return 0, 0

        samples, indices, weights = self.memory.sample(self.batch_size, device)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states).to(device).squeeze()[:, 1, :, :, :] / 255
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(
            -1
        )
        next_states = torch.stack(next_states).to(device).squeeze()[:, 1, :, :, :] / 255
        dones = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(-1)
        weights = weights.unsqueeze(-1)

        (
            predicted_states,
            predicted_next_states,
        ) = self.online_net(states)

        print(f'''State sizes:
------------
Predicted: {predicted_states.size()}
Actual: {states.size()}

              ''')
        # agent_loss = nn.CrossEntropyLoss()

        reconstruction_loss1 = nn.MSELoss()(predicted_states, states)
        reconstruction_loss2 = nn.MSELoss()(predicted_next_states, next_states)
        reconstruction_loss = reconstruction_loss1 + reconstruction_loss2

        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        self.memory.update_priorities(
            indices, reconstruction_loss.detach().cpu().numpy().reshape(-1)
        )
        self.optimizer.step()

        return reconstruction_loss1.item(), reconstruction_loss2.item()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))


# -------------------


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

        fig, axes = plt.subplots(2, 6, figsize=(12, 6))
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
        axes[0,5].imshow(actual.astype(int)[6, :, :])
        axes[0,5].set_title(f'A. {time[0:3]}: Empty')
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
        axes[1,5].imshow(predicted.astype(int)[6, :, :], vmin = 0., vmax = 1.)
        axes[1,5].set_title(f'P. {time[0:3]}: Empty')
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
            state_tensor = state_tensor[:, 1, :, :, :]

            with torch.no_grad():
                decoded1, decoded2 = model.online_net(state_tensor)

            predicted_next_state = decoded2[0].cpu().numpy()
            actual_next_state = next_state.cpu().squeeze().numpy()

            predicted_state = decoded1[0].cpu().numpy()
            actual_state = state.cpu().squeeze().numpy()

            # remove empty layer
            actual_state = actual_state[1, :, :, :]
            actual_next_state = actual_next_state[1, :, :, :]

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
                2,
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
                # Remove empty layer
                state_tensor = state_tensor[:, 1, :, :, :]

                with torch.no_grad():
                    decoded1, decoded2 = model.online_net(state_tensor)

                predicted_next_state = decoded2[0].cpu().numpy()
                actual_next_state = next_state.cpu().squeeze().numpy()

                predicted_state = decoded1[0].cpu().numpy()
                actual_state = state.cpu().squeeze().numpy()

                # remove empty layer
                actual_state = actual_state[1, :, :, :]
                actual_next_state = actual_next_state[1, :, :, :]

                img = plot_states(actual_state, predicted_state, 'current', one_hot, filepath)
                img2 = plot_states(actual_next_state, predicted_next_state, 'next', one_hot, filepath)

                imgs.append(img)
                imgs2.append(img2)

                if turn >= 100:
                    done = 1
                

    imgs[0].save(video_path, format = 'GIF', append_images = imgs[1:], save_all = True, duration = 100, loop = 0)
    imgs2[0].save(video_path2, format = 'GIF', append_images = imgs2[1:], save_all = True, duration = 100, loop = 0)

    # ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)
    # ani.save(video_path, writer="PillowWriter", fps=2)

    # ani2 = animation.ArtistAnimation(fig2, imgs2, interval=50, blit=True, repeat_delay=1000)
    # ani2.save('2_' + video_path, writer="PillowWriter", fps=2)


    # state, action, _, next_state, _ = sample[0]

    # state_tensor = state.unsqueeze(0).to(device).squeeze().unsqueeze(0)
    # # Remove empty layer
    # state_tensor = state_tensor[:, 1, :, :, :]

    # with torch.no_grad():
    #     decoded1, decoded2 = model.online_net(state_tensor)

    # predicted_next_state = decoded2[0].cpu().numpy()
    # actual_next_state = next_state.cpu().squeeze().numpy()

    # predicted_state = decoded1[0].cpu().numpy()
    # actual_state = state.cpu().squeeze().numpy()

    # # remove empty layer
    # actual_state = actual_state[1, :, :, :]
    # actual_next_state = actual_next_state[1, :, :, :]

    # plot_states(actual_state, predicted_state, 'current', one_hot, filepath)
    # plot_states(actual_next_state, predicted_next_state, 'next', one_hot, filepath)

    

