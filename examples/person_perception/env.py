from examples.person_perception.elements import (
    Agent,
    Gem,
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
                    reward = 10
                else:
                    reward = -10
            if color == 1:
                image_color = [0.0, 255.0, 0.0]
                if random.random() < probs[1]:
                    reward = 10
                else:
                    reward = -10
            app = [[0, 0] + image_color + image_color + individuation + [0, 0]]
            info = (person, app, reward, 0, 0)
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

    def softmax(self, x):
        """Compute the softmax of a list of numbers."""
        e_x = np.exp(x - np.max(x))  # subtract max to stabilize
        return e_x / e_x.sum()

    def step(self, models, n):
        predictions = []
        for person in range(n):
            predict, _ = models(torch.tensor(self.appearance[person]).float())
            pred = predict[0][1].detach().numpy()
            predictions.append(float(pred))
        probs = self.softmax(predictions)

        action = np.random.choice(len(probs), p=probs)

        done = 0
        object_info = self.appearance[action]
        reward = self.rewards[action]

        return (
            reward,
            done,
            object_info,
        )


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


class ValueModel(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim=64,
        memory_size=5000,
        learning_rate=0.001,
        num_tau=32,
    ):
        super(ValueModel, self).__init__()
        self.num_tau = num_tau
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_tau)

        self.replay_buffer = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        taus = torch.linspace(0, 1, steps=self.num_tau, device=x.device).view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        quantiles = self.fc4(x)

        # Extract the 25th, 50th, and 75th percentiles
        percentiles = quantiles[
            :,
            [
                int(self.num_tau * 0.1) - 1,
                int(self.num_tau * 0.5) - 1,
                int(self.num_tau * 0.9) - 1,
            ],
        ]
        return percentiles, taus

    def sample(self, num_memories):
        return random.sample(
            self.replay_buffer, min(num_memories, len(self.replay_buffer))
        )

    def learn(self, memories, batch_size=32):
        for _ in range(len(memories) // batch_size):
            batch = random.sample(memories, batch_size)
            states, rewards = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            rewards = (
                torch.tensor(rewards, dtype=torch.float32)
                .view(-1, 1)
                .repeat(1, self.num_tau)
            )

            self.optimizer.zero_grad()
            # Forward pass to get all quantiles, not just the 25th, 50th, and 75th percentiles
            x = torch.relu(self.fc1(states))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            quantiles = self.fc4(x)  # Shape [batch_size, num_tau]

            errors = rewards - quantiles
            huber_loss = torch.where(
                errors.abs() < 1, 0.5 * errors**2, errors.abs() - 0.5
            )
            taus = (
                torch.linspace(0, 1, steps=self.num_tau, device=states.device)
                .view(1, -1)
                .repeat(batch_size, 1)
            )
            quantile_loss = (taus - (errors < 0).float()).abs() * huber_loss
            loss = quantile_loss.mean()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def add_memory(self, state, reward):
        self.replay_buffer.append((state, reward))


env = RPG()
value_model = ValueModel(state_dim=20, memory_size=2000)
rewards = 0
losses = 0
for epoch in range(10000):
    env.generate_trial()
    reward, done, object_info = env.step(value_model, 4)
    rewards += reward
    value_model.add_memory(object_info[0], reward)
    if len(value_model.replay_buffer) > 51:
        memories = value_model.sample(50)
        value_loss = value_model.learn(memories, 50)
        losses = value_loss + losses
    if epoch % 100 == 0:
        print(epoch, rewards / 100, losses / 100)
        rewards = 0
        losses = 0
