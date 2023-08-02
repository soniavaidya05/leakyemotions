import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from examples.RPG3.iRainbow_noCnn import iRainbowModel

# clunky old model from gem

SEED = 1  # Seed for replicating training runs
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# noisy_dueling

device = "cpu"
NETWORK_CONFIG = "noisy_dueling"


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """
    models = []
    models.append(
        iRainbowModel(
            in_channels=9,
            num_filters=20,
            cnn_out_size=2600,
            state_size=torch.tensor(
                [9, 9, 9]
            ),  # this seems to only be reading the first value
            action_size=5,
            network=NETWORK_CONFIG,
            munchausen=False,  # Don't use Munchausen RL loss
            layer_size=100,
            n_hidden_layers=2,
            n_step=1,  # Multistep IQN (rainbow paper uses 3)
            BATCH_SIZE=32,
            BUFFER_SIZE=1024,
            LR=0.0001,  # 0.00025
            TAU=1e-3,  # Soft update parameter
            GAMMA=0.95,  # Discout factor 0.99
            N=32,  # Number of quantiles
            worker=1,  # number of parallel environments
            device=device,
            seed=SEED,
        )
    )  # agent model 1

    return models


class dual_gamble:
    def __init__(
        self,
        num_gambles=1,
        pos_values=[10, 7, 4, 1],
        neg_values=[-10, -7, -4, -1],
        pos_probs=[0.8, 0.6, 0.4, 0.2],
    ):
        self.pos_values = pos_values
        self.neg_values = neg_values
        self.pos_probs = pos_probs
        self.num_gambles = num_gambles

    def generate_trial(self):
        pos1 = np.random.choice(4)
        neg1 = np.random.choice(4)
        prob1 = np.random.choice(4)

        # convert pos, neg, and prob into one hot codes and concatenate them
        pos1_one_hot = np.eye(len(self.pos_values))[pos1]
        neg1_one_hot = np.eye(len(self.neg_values))[neg1]
        prob1_one_hot = np.eye(len(self.pos_probs))[prob1]

        pos1_possible = self.pos_values[pos1]
        neg1_possible = self.neg_values[neg1]
        prob1_possible = self.pos_probs[prob1]

        if random.random() < prob1_possible:
            outcome1 = pos1_possible
        else:
            outcome1 = neg1_possible

        pos2_one_hot = [0, 0, 0, 0]
        neg2_one_hot = [0, 0, 0, 0]
        prob2_one_hot = [0, 0, 0, 0]
        outcome2 = 0

        if self.num_gambles == 2:
            pos2 = np.random.choice(4)
            neg2 = np.random.choice(4)
            prob2 = np.random.choice(4)

            # convert pos, neg, and prob into one hot codes and concatenate them
            pos2_one_hot = np.eye(len(self.pos_values))[pos2]
            neg2_one_hot = np.eye(len(self.neg_values))[neg2]
            prob2_one_hot = np.eye(len(self.pos_probs))[prob2]

            pos2_possible = self.pos_values[pos2]
            neg2_possible = self.neg_values[neg2]
            prob2_possible = self.pos_probs[prob2]

            if random.random() < prob2_possible:
                outcome2 = pos2_possible
            else:
                outcome2 = neg2_possible

        state1 = np.hstack((pos1_one_hot, neg1_one_hot, prob1_one_hot))
        state2 = np.hstack((pos2_one_hot, neg2_one_hot, prob2_one_hot))

        out1 = outcome1
        out2 = outcome2

        counter_balance = np.random.choice(2)
        if counter_balance == 0:
            state = np.hstack((state1, state2))
            out1 = outcome1
            out2 = outcome2
        else:
            state = np.hstack((state2, state1))
            out1 = outcome2
            out2 = outcome1

        state = torch.tensor(state).float()

        return state, out1, out2


env = dual_gamble(num_gambles=2)  # change to 2 for dual gambles
state_dim = 12 * 2  # length of the concatenated one-hot vectors
action_dim = 2  # assuming 2 different actions
# agent = Agent(state_dim, action_dim)


epochs = 50000000

state, outcome1, outcome2 = env.generate_trial()
done = 0
running_rewards = 0

epsilon = 0.999

models = create_models()


losses = 0

for epoch in range(epochs):
    epsilon *= 0.99999
    action = models[0].take_action(state, epsilon)
    action = action[0]

    if action == 0:
        reward = outcome1
    else:
        reward = outcome2

    running_rewards += reward

    prev_state = state
    state, outcome1, outcome2 = env.generate_trial()

    models[0].memory.add(prev_state, action, reward, state, done)

    if epoch > 100 == 0 and epoch % 4 == 0:
        experiences = models[0].memory.sample()
        loss = models[0].learn(experiences)
        losses = losses + loss

    if epoch % 1000 == 0:
        print(epoch, losses, running_rewards, epsilon)
        running_rewards = 0
        losses = 0
