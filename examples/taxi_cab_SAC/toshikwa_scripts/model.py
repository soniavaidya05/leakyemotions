import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
#from abc import ABC, abstractmethod
import torch.optim as optim
import numpy as np

from examples.taxi_cab_SAC.memory.per import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from examples.taxi_cab_SAC.toshikwa_scripts.utils import update_params, RunningMeanStats


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, in_channels, num_filters):
        super(DQNBase, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=1
        )
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)

    def forward(self, x):
        x = x / 255  # note, a better normalization should be applied
        y1 = F.relu(self.conv_layer1(x))
        y2 = self.avg_pool(y1)  # ave pool is intentional (like a count)
        y2 = torch.flatten(y2, 1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1, y2), 1)
        return y


class QNetwork(BaseNetwork):

    def __init__(self, in_channels, num_filters, in_size, hid_size1, hid_size2, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(in_channels, num_filters)
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=2,
            batch_first=True,
        )
        self.al1 = nn.Linear(hid_size1, hid_size1) 
        self.al2 = nn.Linear(hid_size1, hid_size2)
        self.al3 = nn.Linear(hid_size2, num_actions)

        self.vl1 = nn.Linear(hid_size1, hid_size1) 
        self.vl2 = nn.Linear(hid_size1, hid_size2)
        self.vl3 = nn.Linear(hid_size2, 1)


        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(in_size, hid_size1), # was 7 * 7 * 64, 512
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size1, hid_size1),
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size1, num_actions)) # was 512
        else:
            self.a_head = nn.Sequential(
                nn.Linear(in_size, hid_size1),
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size1, hid_size1),
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(405, hid_size1),
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size1, hid_size1),
                nn.ReLU(inplace=True),
                nn.Dropout(.2),
                nn.Linear(hid_size1, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            batch_size, timesteps, C, H, W = states.size()
            c_in = states.view(batch_size * timesteps, C, H, W)
            c_out = self.conv(c_in)
            r_in = c_out.view(batch_size, timesteps, -1)

        if not self.dueling_net:
            a_r_out, (a_h_n, a_h_c) = self.rnn(r_in)
            a = F.relu(self.al1(a_r_out[:, -1, :])) # what is this was lr = .001
            a = F.relu(self.al2(a)) # and this is lr = .0011 (a small bit more)
            a = self.al3(a)
            return a
        else:
            a_r_out, (a_h_n, a_h_c) = self.rnn(r_in)
            a = F.relu(self.al1(a_r_out[:, -1, :])) # what is this was lr = .001
            a = F.relu(self.al2(a)) # and this is lr = .0011 (a small bit more)
            a = self.al3(a)

            v_r_out, (v_h_n, v_h_c) = self.rnn(r_in)
            v = F.relu(self.vl1(v_r_out[:, -1, :])) # what is this was lr = .001
            v = F.relu(self.vl2(v)) # and this is lr = .0011 (a small bit more)
            v = self.vl3(v)

            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, in_channels, num_filters, in_size, hid_size1, hid_size2, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(in_channels, num_filters, in_size, hid_size1, hid_size2, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(in_channels, num_filters, in_size, hid_size2, hid_size2, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CategoricalPolicy(BaseNetwork):

    def __init__(self, in_channels, num_filters, in_size, hid_size1, hid_size2, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(in_channels, num_filters)

        self.rnn = nn.LSTM(
        input_size=in_size,
        hidden_size=hid_size1,
        num_layers=2,
        batch_first=True,
        )
        self.al1 = nn.Linear(hid_size1, hid_size1) 
        self.al2 = nn.Linear(hid_size1, hid_size2)
        self.al3 = nn.Linear(hid_size2, num_actions)

        self.vl1 = nn.Linear(hid_size1, hid_size1) 
        self.vl2 = nn.Linear(hid_size1, hid_size2)
        self.vl3 = nn.Linear(hid_size2, num_actions)


        self.head = nn.Sequential(
            nn.Linear(in_size, hid_size1),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size1, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            batch_size, timesteps, C, H, W = states.size()
            c_in = states.view(batch_size * timesteps, C, H, W)
            c_out = self.conv(c_in)
            r_in = c_out.view(batch_size, timesteps, -1)


        v_r_out, (v_h_n, v_h_c) = self.rnn(r_in)
        v = F.relu(self.vl1(v_r_out[:, -1, :])) # what is this was lr = .001
        v = F.relu(self.vl2(v)) # and this is lr = .0011 (a small bit more)
        action_logits = self.al3(v)


        #action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=-1, keepdim=True)  # previously dim = 1, changing to -1
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            batch_size, timesteps, C, H, W = states.size()
            c_in = states.view(batch_size * timesteps, C, H, W)
            c_out = self.conv(c_in)
            r_in = c_out.view(batch_size, timesteps, -1)

        v_r_out, (v_h_n, v_h_c) = self.rnn(r_in)
        v = F.relu(self.vl1(v_r_out[:, -1, :])) # what is this was lr = .001
        v = F.relu(self.vl2(v)) # and this is lr = .0011 (a small bit more)
        v = self.al3(v)



        action_probs = F.softmax(v, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

class SAC():

    kind = "soft_actor_critic"  # class variable shared by all instances

    def __init__(
        self,
        in_channels,
        num_filters,
        lr,
        replay_size,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        priority_replay=True,
        device="cpu",
        state_shape = (3,4,9,9),
        gamma = .9,
        use_per=True,
        num_steps = 100, # the next four are made up
        start_steps = 10,
        update_interval = 1,
        multi_step = 1,
        target_entropy_ratio=0.98
    ):
        super().__init__()
        #self.gamma = gamma
        #self.state_shape = state_shape
        self.device = device
        self.modeltype = "soft_actor_critic"
        # Define networks.
        self.policy = CategoricalPolicy(in_channels, num_filters, in_size, hid_size1, hid_size2, out_size).to(self.device)
        self.online_critic = TwinnedQNetwork(in_channels, num_filters, in_size, hid_size1, hid_size2,out_size).to(device=self.device)
        self.target_critic = TwinnedQNetwork(in_channels, num_filters, in_size, hid_size1, hid_size2,out_size).to(device=self.device).eval()

        beta_steps = (num_steps - start_steps) / update_interval
        self.memory = LazyPrioritizedMultiStepMemory(
            capacity=replay_size,
            state_shape=state_shape,
            device=self.device, gamma=gamma, multi_step=multi_step,
            beta_steps=beta_steps)
        self.use_per = use_per
        self.learning_steps = 0
        self.batch_size = 32 # this is because the CNN is messing up
        self.alpha = 0.6
        self.gamma_n = .9
        self.out_size = out_size

        #disable_gradients(self.target_critic)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.out_size) * target_entropy_ratio 

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def learn(self):
        #assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
        #    hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss


    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long()) 
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

        #if self.learning_steps % self.log_interval == 0:
        #    self.writer.add_scalar(
        #        'loss/Q1', q1_loss.detach().item(),
        #        self.learning_steps)
        #    self.writer.add_scalar(
        #        'loss/Q2', q2_loss.detach().item(),
        #        self.learning_steps)
        #    self.writer.add_scalar(
        #        'loss/policy', policy_loss.detach().item(),
        #        self.learning_steps)
        #    self.writer.add_scalar(
        #        'loss/alpha', entropy_loss.detach().item(),
        #        self.learning_steps)
        #    self.writer.add_scalar(
        #       'stats/alpha', self.alpha.detach().item(),
        #        self.learning_steps)
        #    self.writer.add_scalar(
        #        'stats/mean_Q1', mean_q1, self.learning_steps)
        #    self.writer.add_scalar(
        #        'stats/mean_Q2', mean_q2, self.learning_steps)
        #    self.writer.add_scalar(
        #        'stats/entropy', entropies.detach().mean().item(),
        #        self.learning_steps)

    #@abstractmethod
    #def save_models(self, save_dir):
    #    if not os.path.exists(save_dir):
    #        os.makedirs(save_dir)

    #def __del__(self):
    #    self.env.close()
    #    self.test_env.close()
    #    self.writer.close()

    




     





