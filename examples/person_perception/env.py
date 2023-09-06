from examples.person_perception.elements import (
    Agent,
    Gem,
    EmptyObject,
    Wall,
)

import numpy as np
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.models.perception_singlePixel_categories import agent_visualfield
import random

from gem.utils import find_moveables, find_instance
import torch


class RPG:
    def __init__(
        self,
        group_probs=[0.50, 0.50],
        num_people=150,
    ):
        self.group_probs = group_probs
        self.num_people = num_people
        self.person_list = []
        self.create_people(num_people, group_probs)

    def create_people(self, num_people, probs):
        for person in range(num_people):
            binary_indiv = False
            if binary_indiv:
                individuation = [
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    np.random.choice([0, 1]) * 255.0,
                    0,
                    0,
                ]
            else:
                individuation = [
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    random.random() * 255.0,
                    0,
                    0,
                ]

            color = np.random.choice([0, 1])
            if color == 0:
                image_color = [255.0, 0.0, 0.0]
                if random.random() < probs[0]:
                    rock = 1
                    wood = 0
                else:
                    wood = 1
                    rock = 0
            if color == 1:
                image_color = [0.0, 255.0, 0.0]
                if random.random() < probs[1]:
                    rock = 1
                    wood = 0
                else:
                    wood = 1
                    rock = 0
            app = [[0, 0] + image_color + image_color + individuation + [0, 0]]
            info = (person, app, [wood, rock], 0, 0)
            self.person_list.append(info)

    def reset_env(self, num_people, probs):
        """
        Resets the environment and repopulates it
        """

        self.create_people(num_people, probs)

    def generate_trial(self, n=4):
        random_numbers = random.sample(range(len(self.person_list)), n)
        self.appearance = {i: None for i in range(n)}
        self.rewards = {i: None for i in range(n)}
        for person in range(n):
            self.appearance[person] = self.person_list[random_numbers[person]][1]
            self.rewards[person] = self.person_list[random_numbers[person]][2]

    def step(self, models, n):
        predictions = []
        for person in range(n):
            predict = models(torch.tensor(self.appearance[person]).float())
            predictions.append(predict.detach().numpy())
        print(predictions)
        # action = softmax(predictions)
        # return action


# all below needs to be moved


import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video

import torch.optim as optim

import time
import numpy as np
import random
import torch

from collections import deque, namedtuple
from scipy.spatial import distance

from datetime import datetime

from sklearn.neighbors import NearestNeighbors


class ResourceModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
    ):
        super(ResourceModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 3)  # Three outputs for three classes

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        probabilities = torch.softmax(self.fc4(x), dim=-1)
        return probabilities

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32, class_weights=False):
        if class_weights:
            # Calculate class weights
            all_outcomes = [outcome for _, outcome in self.replay_buffer]
            num_samples = len(all_outcomes)
            class_counts = [sum([out[i] for out in all_outcomes]) for i in range(3)]

            # Adding a small epsilon to prevent division by zero
            epsilon = 1e-10
            class_weights = torch.tensor(
                [(num_samples / (count + epsilon)) for count in class_counts]
            ).to(torch.float32)

            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Weighted Cross-Entropy Loss
                loss = F.cross_entropy(
                    probabilities, torch.argmax(targets, dim=1), weight=class_weights
                )

                loss.backward()
                self.optimizer.step()

        else:
            for _ in range(len(memories) // batch_size):
                batch = random.sample(memories, batch_size)
                states, targets = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Cross-Entropy Loss without weights
                loss = F.cross_entropy(probabilities, torch.argmax(targets, dim=1))

                loss.backward()
                self.optimizer.step()

        return loss.item()

    def add_memory(self, state, outcome):
        self.replay_buffer.append((state, outcome))


env = RPG()
env.generate_trial()
resource_model = ResourceModel(state_dim=20, memory_size=2000)
env.step(resource_model, 4)
