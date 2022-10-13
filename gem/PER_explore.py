from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
)
from gem.environment.elements.AI_econ_elements import (
    Agent,
    Wood,
    Stone,
    House,
    EmptyObject,
    Wall,
)
from models.cnn_lstm_dqn import Model_CNN_LSTM_DQN
from gemworld.AI_econ_world import AI_Econ
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from DQN_utils import save_models, load_models, make_video2

import random

import numpy as np

import torch

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"


models = load_models(save_dir, "AI_econ_test28")
make_video2("test_new_priority", save_dir, models, 30, env)


replay = models[0].replay
losses = models[0].surprise(replay, 0.7)


fig = plt.figure()
ax = plt.axes()
plt.title("Line graph")
plt.plot(losses.numpy(), color="red")

sample_indices, importance_normalized = models[0].priority_sample(
    losses, sample_size=256, alpha_scaling=0.7, offset=0.1
)

fig = plt.figure()
ax = plt.axes()
plt.title("Line graph")
plt.plot(importance_normalized, color="red")

im_losses = losses[sample_indices].numpy()
im_losses = im_losses / np.max(im_losses)

plt.plot(importance_normalized, color="red")
plt.plot(im_losses, color="blue")


plt.scatter(importance_normalized, im_losses)

b = 0.3
plt.plot(importance_normalized**b, color="red")
plt.scatter(importance_normalized**b, im_losses)


minibatch = [models[0].replay[i] for i in sample_indices]

state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

Q1 = models[0].model1(state1_batch)
with torch.no_grad():
    Q2 = models[0].model2(state2_batch)

gamma = 0.9

Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2.detach(), dim=1)[0])
X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

loss = (torch.FloatTensor(importance_normalized) * F.mse_loss(X, Y)).mean()

X - Y.detach()


loss1 = torch.FloatTensor(importance_normalized) * F.mse_loss(X, Y)
loss2 = torch.FloatTensor(importance_normalized) * ((X - Y.detach()) ** 2)

plt.scatter(loss1.detach().numpy(), loss2.detach().numpy())
plt.scatter(loss1.detach().numpy(), importance_normalized)
plt.scatter(loss1.detach().numpy(), im_losses)
plt.scatter(loss2.detach().numpy(), importance_normalized)
plt.scatter(loss2.detach().numpy(), im_losses)

plt.plot(loss2.detach().numpy(), color="red")
plt.plot(im_losses, color="red")


im_losses
