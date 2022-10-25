import math
import random
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
import gym


#### from https://github.com/rtharungowda/Soft-Actor-Critic-Pytorch/blob/master/pendulumn/train.py


#### This is for continous action space only not discrete

class ReplayBuffer(object):
	def __init__(self,capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self,state,action,reward,next_state,done):
		if(len(self.buffer)<self.capacity):
			self.buffer.append(None)

		self.buffer[int(self.position)] = (state,action,reward,next_state,done)
		self.position = (self.position+1)%self.capacity

	def sample(self,batch_size):
		batch = iter(random.sample(self.buffer,batch_size))
		# print(batch[0])
		state,action,reward,next_state,done = map(np.stack,zip(*batch)) #batch[:][0],batch[:][1],batch[:][2],batch_size[:][3],batch[:][4] 
		return state,action,reward,next_state,done

	def __len__(self):
		return len(self.buffer)


class ValueNetwork(nn.Module):
	def __init__(self,state_dim,hidden_dim,init_w=3e-3):
		super(ValueNetwork,self).__init__()
		self.linear1 = nn.Linear(state_dim,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,hidden_dim)
		self.linear3 = nn.Linear(hidden_dim,1)

		self.linear3.weight.data.uniform_(-init_w,init_w)
		self.linear3.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class SoftQNetwork(nn.Module):
	def __init__(self,state_dim,action_dim,hidden_size,init_w=3e-3):
		super(SoftQNetwork,self).__init__()
		self.linear1 = nn.Linear(state_dim+action_dim,hidden_size)
		self.linear2 = nn.Linear(hidden_size,hidden_size)
		self.linear3 = nn.Linear(hidden_size,1)

		self.linear3.weight.data.uniform_(-init_w,init_w)
		self.linear3.bias.data.uniform_(-init_w,init_w)

	def forward(self,state,action):
		x = torch.cat((state,action),dim=1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class PolicyNetwork(nn.Module):
	def __init__(self,state_dim,action_dim,hidden_dim,init_w=3e-3,log_std_min=-20,log_std_max=2):
		super(PolicyNetwork,self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.linear1 = nn.Linear(state_dim,hidden_dim)
		self.linear2 = nn.Linear(hidden_dim,hidden_dim)

		self.mean_linear = nn.Linear(hidden_dim,action_dim)
		self.mean_linear.weight.data.uniform_(-init_w,init_w)
		self.mean_linear.bias.data.uniform_(-init_w,init_w)

		self.log_std_linear = nn.Linear(hidden_dim,action_dim)
		self.log_std_linear.weight.data.uniform_(-init_w,init_w)
		self.log_std_linear.bias.data.uniform_(-init_w,init_w)

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

		return mean,log_std

	def evaluate(self,state,epsilon=1e-6):

		mean,log_std = self.forward(state)
		std = log_std.exp() #exponentiating to get +ve std
		normal = Normal(0,1)
		z = normal.sample()
		action = torch.tanh(mean+std*z)
		#https://pytorch.org/docs/stable/distributions.html
		#log_prob to create a differentiable loss function
		#Normal(mean,sd)l.og_prob(a) finds the normal dist around the passed value 'a'
		log_prob = Normal(mean,std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
		return action, log_prob, z, mean, log_std

	def get_action(self,state):
		state = torch.FloatTensor(state).unsqueeze(0)
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean,std)
		z = normal.sample()
		action = torch.tanh(z)
		return action[0]





# other stuff


action_dim = 4 #env.action_space.shape[0]
state_dim  = 10 #env.observation_space.shape[0]
hidden_dim = 256

#value function
value_net = ValueNetwork(state_dim,hidden_dim)
target_value_net = ValueNetwork(state_dim,hidden_dim)

# using two q func to minimise overestimation 
# and choose the minimum of the two nets
soft_q_net1 = SoftQNetwork(state_dim,action_dim,hidden_dim)
soft_q_net2 = SoftQNetwork(state_dim,action_dim,hidden_dim)

#policy function
policy_net = PolicyNetwork(state_dim,action_dim,hidden_dim)

for target_param, param in zip(target_value_net.parameters(),value_net.parameters()):
	target_param.data.copy_(param.data)

#loss
value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

lr = 3e-4

#optimizer
value_optimizer = optim.Adam(value_net.parameters(),lr=lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(),lr=lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(),lr=lr)
policy_opimizer = optim.Adam(policy_net.parameters(),lr=lr)

#buffer
replay_buffer_size = 10e6
replay_buffer = ReplayBuffer(replay_buffer_size)






#### UPDATE FUNCTION



#simulation on aggregated observations
def update(batch_size,gamma=0.99,soft_tau=1e-2):

	state, action, reward, next_state, done = replay_buffer.sample(batch_size)
	state = torch.FloatTensor(state)
	next_state = torch.FloatTensor(next_state)
	action = torch.FloatTensor(action)
	reward = torch.FloatTensor(reward)
	reward = torch.unsqueeze(reward,dim=1)
	done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

	predicted_q_value1 = soft_q_net1(state, action)
	predicted_q_value2 = soft_q_net2(state, action)

	predicted_value = value_net(state)

	new_action, log_prob ,_,_,_ = policy_net.evaluate(state)

	#----Training Q Function----
	target_value = target_value_net(next_state)
	target_q_value = reward+(1-done)*gamma*target_value 

	# loss = (Q(s_t,a_t) - (r + gamma*V(s_t+1)) )**2
	q_value_loss1 = soft_q_criterion1(predicted_q_value1,target_q_value.detach())
	q_value_loss2 = soft_q_criterion2(predicted_q_value2,target_q_value.detach())
	

	soft_q_optimizer1.zero_grad()
	q_value_loss1.backward()
	soft_q_optimizer1.step()

	soft_q_optimizer2.zero_grad()
	q_value_loss2.backward()
	soft_q_optimizer2.step()


	#----Training Value Function----
	predicted_new_q_value = torch.min(soft_q_net1(state,new_action),soft_q_net2(state,new_action))
	target_value_func = predicted_new_q_value - log_prob

	#loss = (V(s_t) - ( Q(s_t,a_t) + H(pi(:,s_t)) ))**2  --H(pi(:,s_t)) = -log(pi(:,s_t))--
	value_loss = value_criterion(predicted_value,target_value_func.detach())

	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()


	#----Training Policy Function----
	#maximise (Q(s_t,a_t) + H(pi(:,s_t)) )--> (Q(s_t,a_t) - log(pi(:,s_t)) 
	#minimise (log(pi:,s_t) - Q(s_t,a_t))
	policy_loss = (log_prob - predicted_new_q_value).mean()

	policy_opimizer.zero_grad()
	policy_loss.backward()
	policy_opimizer.step()

	#Update target network parameters
	for target_param, param in zip(target_value_net.parameters(),value_net.parameters()):
		target_param.data.copy_(soft_tau*param + (1-soft_tau)*target_param)


### FROM THE EXAMPLE

max_frames  = 40000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 32



### TRY IN TAXI CAB

# from tkinter.tix import Tree
from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_agents,
    find_instance,
)
from examples.taxi_cab.elements import (
    TaxiCab,
    EmptyObject,
    Wall,
    Passenger,
)

from  examples.taxi_cab_AC.env import TaxiCabEnv
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video
import torch

import random

# save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "/Users/ethan/gem_output/"
save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

device = "cpu"
print(device)

world_size = 10

trainable_models = [0]
sync_freq = 500
modelUpdate_freq = 25
epsilon = 0.99

turn = 1

#models = create_models()
env = TaxiCabEnv(
    height=world_size,
    width=world_size,
    layers=1,
    defaultObject=EmptyObject,
)

losses = 0
game_points = [0, 0]
epochs = 100
max_turns = 100
world_size = 10
env.reset_env(
    height=world_size,
    width=world_size,
    layers=1,
)

from examples.taxi_cab_AC.cnn_lstm_AC import Model_CNN_LSTM_AC
def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    models.append(
        Model_CNN_LSTM_AC(
            in_channels = 4,
            numFilters = 5, 
            lr = .001, 
            replay_size = 3, 
            in_size = 650, 
            hid_size1 = 75, 
            hid_size2 = 30, 
            out_size = 4
            # note, need to add device and maybe in_channels
    )
    )  # taxi model

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
        models[model].model2.to(device)
    # currently the AC has two models, which it doesn't need
    # and is wasting space
    
    return models

models = create_models()

for turn in range(1000):
    agentList = find_instance(env.world, "neural_network")
    random.shuffle(agentList)

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(3)

    loc = agentList[0]

    env.world[loc].reward = 0




    if env.world[loc].action_type == "neural_network":

        holdObject = env.world[loc]
        device = "cpu"
        #state = env.pov(loc, inventory=[holdObject.has_passenger], layers=[0])
        state = torch.rand(10).float()
        next_state = torch.rand(10).float()
        reward = torch.rand(1).float()
        action_ch = np.random.choice([0,1,2,3])
        action = [0,0,0,0]
        action[action_ch] = 1
        action = torch.tensor(action).float()
        #action = torch.tensor(np.random.choice([0,1,2,3])).float()
        done = torch.tensor(np.random.choice([0,0,0,1])).float()


        #action, logprob, value = models[holdObject.policy].take_action([state.to(device), epsilon])
        #"""
        #Updates the world given an action
        #"""
        #(
        #    env.world,
        #    reward,
        #    next_state,
        #    done,
        #    new_loc,
        #) = holdObject.transition(env, models, action, loc)
    #else:
        #pass




    replay_buffer.push(state,action,reward,next_state,done)

    if len(replay_buffer) > batch_size:
	    #run simulation
        update(batch_size)



