from sklearn.neighbors import NearestNeighbors
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


def compute_weighted_average(
    state, memories, similarity_decay_rate=1, time_decay_rate=1
):
    if not memories:
        return 0

    memory_states, rewards = zip(*memories)
    memory_states = np.array(memory_states)
    state = np.array(state)

    # Compute Euclidean distances
    distances = np.linalg.norm(memory_states - state, axis=1)
    max_distance = np.max(distances) if distances.size else 1

    # Compute similarity weights with exponential decay
    similarity_weights = (
        np.exp(-distances / max_distance * similarity_decay_rate)
        if max_distance != 0
        else np.ones_like(distances)
    )

    # Compute time weights with exponential decay
    N = len(memories)
    time_weights = np.exp(-np.arange(N) / (N - 1) * time_decay_rate)

    # Combine the weights
    weights = similarity_weights * time_weights

    # Compute the weighted sum
    weighted_sum = np.dot(weights, rewards)
    total_weight = np.sum(weights)

    return weighted_sum / total_weight if total_weight != 0 else 0


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
                states = torch.tensor(np.array(states), dtype=torch.float32)
                targets = torch.tensor(np.array(targets), dtype=torch.float32)

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
                states = torch.tensor(np.array(states), dtype=torch.float32)
                targets = torch.tensor(np.array(targets), dtype=torch.float32)

                self.optimizer.zero_grad()
                probabilities = self.forward(states)

                # Cross-Entropy Loss without weights
                loss = F.cross_entropy(probabilities, torch.argmax(targets, dim=1))

                loss.backward()
                self.optimizer.step()

        return loss.item()

    def add_memory(self, state, outcome):
        self.replay_buffer.append((state, outcome))


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
            states = torch.tensor(np.array(states), dtype=torch.float32)
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


def eval_attitude_model(value_model):
    atts = []
    s = torch.zeros(7)
    r = value_model(s)
    atts.append(round(r.item(), 2))
    for a in range(7):
        s = torch.zeros(7)
        s[a] = 255.0
        r = value_model(s)
        atts.append(round(r.item(), 2))
    return atts


class EWAModel:

    """
    Model class for the Episodic memory model with search and reWeighting (EWA)
    """

    def __init__(
        self, mem_len, state_knn_len, episodic_decay_rate, similarity_decay_rate
    ):
        self.memory = deque(maxlen=mem_len)
        self.state_knn = NearestNeighbors(n_neighbors=state_knn_len)
        self.episodic_decay_rate = episodic_decay_rate
        self.similarity_decay_rate = similarity_decay_rate

    def get_states(self):
        return torch.tensor(np.array([obj_mem[0] for obj_mem in self.memory]))

    def fit(self):
        self.state_knn.fit([obj_mem[0] for obj_mem in self.memory])

    def k_most_similar_recent_states(self, state, all_states, k=5, USE_KNN_MODEL=True):
        if all_states is None:
            all_states = self.get_states()

        if USE_KNN_MODEL:
            # Get the indices of the k most similar states (without selecting them yet)
            state = state.cpu().detach().numpy().reshape(1, -1)
            k_indices = self.state_knn.kneighbors(
                state, n_neighbors=k, return_distance=False
            )[0]

        else:
            # Perform a brute-force search for the k most similar states
            # distances = [distance.euclidean(state, memory[0]) for memory in memories]
            # k_indices = np.argsort(distances)[:k]

            # let's try another way using torch operations...

            # Calculate the squared Euclidean distance
            squared_diff = torch.sum((all_states - state) ** 2, dim=1)
            # Take the square root to get the Euclidean distance
            distance = torch.sqrt(squared_diff)
            # Argsort and take top-k
            k_indices = torch.argsort(distance, dim=0)[:k]

        # Gather the k most similar memories based on the indices, preserving the order
        most_similar_memories = [self.memory[i] for i in k_indices]

        return most_similar_memories


# ---------------------------- #
# In game evaluation functions #
# ---------------------------- #


def evaluate(
    env,
    condition,
    value_model,
    resource_model,
    ewa_model,
    epoch,
    loc=None,
    testing=False,
):
    """
    Use the chosen model to update the appearance of objects in the environment
    env: The environment
    condition: the attitude model condition
    value_model: the value model
    resource_model: the resource model
    loc: (Not yet implemented) the location of the agent, so appearances are updated only within the agent's field of view.
    Requires the use of evaluation per turn rather than per game.
    testing: (Optional) whether to test the attitude model and print out its outputs
    """

    # First, zero out the last three digits of the objects in the environment
    for i in range(env.height):
        for j in range(env.width):
            # Put zeroes in the last three values
            env.world[i, j, 0].appearance.put([-3, -2, -1], 0.0)

    # Get the object memory states for EWA model
    if "EWA" in condition:
        all_states = ewa_model.get_states()

    # Get environment points to update
    all_is = [i for i in range(env.height)]
    all_js = [j for j in range(env.width)]

    # If using the agent loc, then update only the visual field
    if loc is not None:
        vision = env.world[loc].vision
        all_is = [loc[0] + i - vision for i in range(env.height)]
        all_js = [loc[1] + i - vision for i in range(env.width)]

    # Then, choose condition

    for i in all_is:
        for j in all_js:
            # Values are only computed within the environment parameters (only relevant when loc is passed in)
            if i >= 0 and j >= 0 and i < env.height and j < env.width:
                # Get values (0, len - 3) of the appearance to build an attitude
                object_state = torch.tensor(
                    np.array(env.world[i, j, 0].appearance[:-3])
                ).float()

                # ----------------- #
                # Implicit attitude #
                # ----------------- #
                if "implicit" in condition and epoch > 2:
                    # Get estimated reward associated with the object state
                    rs, _ = value_model(object_state.unsqueeze(0))
                    r = rs[0][1]

                    # Assign the reward value to the appearance of the object
                    env.world[i, j, 0].appearance[-2] = r.item() * 255

                # ----------------- #
                # Resource learning #
                # ----------------- #
                elif "tree_rocks" in condition and epoch > 2:
                    # Predict the resource distribution of the agent
                    predict = resource_model(object_state)
                    predict = predict.detach().numpy()

                    # Put the prediction values into the last three values
                    assert len(predict) == 3
                    env.world[i, j, 0].appearance.put([-3, -2, -1], predict * 255)

                # ------------------------- #
                # Episodic memory w/ search #
                # ------------------------- #
                elif "EWA" in condition and epoch > 10:
                    # Get the k-nearest-neighbours
                    mems = ewa_model.k_most_similar_recent_states(
                        state=object_state, all_states=all_states, k=10
                    )

                    env.world[i, j, 0].appearance[-1] = (
                        compute_weighted_average(
                            object_state,
                            mems,
                            similarity_decay_rate=ewa_model.similarity_decay_rate,
                            time_decay_rate=ewa_model.episodic_decay_rate,
                        )
                        * 255
                    )

                else:
                    # No attitude model, nothing happens
                    # TODO: Other models to be implemented
                    pass
