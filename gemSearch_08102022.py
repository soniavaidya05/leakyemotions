#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.visualization import make_lupton_rgb

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import copy

import pickle

from collections import deque


# In[2]:


# some useful functions

def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec

def updateEpsilon(epoch, epsilon, turn):  
    if epsilon > 0.1:
        epsilon -= (1/(turn))

    if epsilon > 0.2: 
        if epoch > 1000 and epoch%10000 == 0:
            epsilon -= .1
    return epsilon

def findMoveables(world): 
  moveList = []
  for i in range(world.shape[0]):
    for j in range(world.shape[0]):
      if world[i,j,0].static == 0:
        moveList.append([i,j])
  return moveList

def findTrainables(world): 
  moveList = []
  for i in range(world.shape[0]):
    for j in range(world.shape[0]):
      if world[i,j,0].trainable == 1:
        moveList.append([i,j])
  return moveList

def updateExperiences(world, models, multipleImages = False): # create the updated end of round experience memories
     moveList = findMoveables(world)
     for i, j in moveList:
         img = agentVisualField(world, (i,j), 4)
         world[i,j,0].CurrExp.nextState = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
         if multipleImages == True:
             world[i,j,0].CurrExp.nextState = torch.cat([world[i,j,0].CurrExp.nextState, world[i,j,0].CurrExp.state[:,0:3,:,:], world[i,j,0].memories[2]], dim = 1)
         exp = (world[i,j,0].CurrExp.state, 
                torch.tensor(world[i,j,0].CurrExp.action).float(), 
                torch.tensor(world[i,j,0].CurrExp.reward).float(), 
                world[i,j,0].CurrExp.nextState, 
                torch.tensor(world[i,j,0].CurrExp.done).float())
         #print(world[i,j,0].CurrExp.state)

         models[world[i,j,0].policy].replayDQN.append(exp)
         world[i,j,0].memories.append(exp)
         world[i,j,0].currentExp_reward = 0
#
#     # will want to reset the memories here as well for efficiency and so we can add rewards to a zero
     return models

# replay memory class

class Memory():
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode):
        self.memory.append(epsiode)

    # get multiple sequences of expereicnes from multiple episodes (stories) (each sequence from a distinct episode)
    def get_batch(self, bsize, time_step):
        sampled_episodes = random.sample(self.memory, bsize)
        batches = []
        for episode in sampled_episodes:
            while len(episode) + 1 - time_step < 1:
                episode = random.sample(self.memory, 1)[0]
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batches.append(episode[point:point + time_step])

        return batches

# current Experiences class
    
class CurrExp():
    def __init__(self):
        self.state = []
        self.reward = 0
        self.action = 0
        self.nextState = []
        self.done = 0
        



# build Double DQN with LSTM CNN neural net models that be be used

class DQN_LSTM_CNN(nn.Module):
  def __init__(self, in_size, hid_size, out_size, value):
    super().__init__()
    self.numFilters = value
    self.h_size = hid_size 
    self.in_size = in_size 
    self.out_size = out_size
    self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels= self.numFilters, kernel_size=1)
    self.lstm = nn.LSTM(in_size, hid_size, batch_first=True)
    self.l1 = nn.Linear(hid_size, hid_size)
    self.l2 = nn.Linear(hid_size, 150)
    self.l3 = nn.Linear(150, 4)
    self.relu = nn.ReLU()
    self.avg_pool = nn.MaxPool2d(3, 1, padding=0)
    self.dropout = nn.Dropout(0.1)

  
  def forward(self, bsize, input, time_step, hidden_state, cell_state):
    input = input.reshape(bsize* time_step, 3, 9, 9)
    y1 = self.conv_layer1(input/255)
    y2 = self.avg_pool(y1)
    y1 = torch.flatten(y1, 1)
    y2 = torch.flatten(y2,1)
    y = torch.cat((y1,y2),1)

    input = y.reshape(bsize, time_step, self.in_size)

    lstm_out, (h_n, c_n)= self.lstm(input, (hidden_state, cell_state))
    lstm_out = lstm_out[:, time_step - 1, :]
    mlp1 = self.dropout(self.relu(self.l1(lstm_out)))
    mlp2 = self.dropout(self.relu(self.l2(mlp1)))
    result = self.l3(mlp2)
    return result, (h_n, c_n)

  def init_hidden_states(self, bsize):
    h = torch.zeros(1, bsize, self.h_size).float()
    c = torch.zeros(1, bsize, self.h_size).float()
    return h, c

#LSTM model class 
class lstm_cnn_modelClass:

    kind = 'modelDQN_LSTM_CNN'                    # class variable shared by all instances

    def __init__(self, value, lr, replaySize, insize, outsize, hidsize):
        self.modeltype = 'lstm_cnn'  
        self.model1 = DQN_LSTM_CNN(insize, hidsize, outsize, value)
        self.model2 = DQN_LSTM_CNN(insize, hidsize, outsize, value)
        self.optimizer = torch.optim.Adam(self.model1.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        self.training_tstep = 5  # I assume that we will want to read this in
        self.replay = Memory(replaySize)
        self.epoch_mem = [] 
        self.h_state = torch.zeros(1, 1, hidsize).float() # note manually entered 1 fore bsize
        self.c_state = torch.zeros(1, 1, hidsize).float()
        self.h_size = hidsize
        self.sm = nn.Softmax(dim=1)

        
    def init_hidden_states(self, bsize):
        self.h = torch.zeros(1, bsize, self.h_size).float()
        self.c = torch.zeros(1, bsize, self.h_size).float()
        #return h, c

    def takeAction(self, inputs):
        world, i, j, vision, epsilon = inputs
        img = agentVisualField(world, (i,j), 4)
        input = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
  
        Q, (h_n, c_n) = self.model1(bsize=1, input=input, time_step=1, 
                                           hidden_state=self.h_state,
                                           cell_state=self.c_state) ##
        self.h_state = h_n ##  we should be able to save the current hidden states in the model, no?
        self.c_state = c_n ##  or does this lead to problems elsewhere?
        p = self.sm(Q).detach().numpy()[0]

        if epsilon > .5:
            if (random.random() < epsilon):
                action = np.random.randint(0,len(p))
            else:
                action = np.argmax(Q.detach().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p = p)

            
        return action, input
    
    def training(self, batch_size, gamma):

      loss = torch.tensor(0.)

      if len(self.replay.memory) > batch_size:  ##
        # print("got here")
        TIME_STEP = self.training_tstep
        h_n, c_n = self.model1.init_hidden_states(bsize=batch_size) ##

        minibatch = self.replay.get_batch(bsize=batch_size, time_step=TIME_STEP)

        current_states = []
        acts = []
        rewards = []
        next_states = []
        done_val = []

        for b in minibatch:
            cs, ac, rw, ns, dones = [], [], [], [], []
            for step in b:
                cs.append(step[0])
                ac.append(step[1])
                rw.append(step[2])
                ns.append(step[3])
                dones.append(step[4])
            current_states.append(torch.stack(cs))
            acts.append(ac)
            rewards.append(rw)
            next_states.append(torch.stack(ns))
            done_val.append(dones)

        action_batch = torch.Tensor(acts)
        reward_batch = torch.Tensor(rewards)
        done_batch = torch.Tensor(done_val)
        done_batch = done_batch[:, TIME_STEP - 1]
        state1_batch = torch.stack(current_states)
        state2_batch = torch.stack(next_states)

        Q1, _ = self.model1(bsize=batch_size, input=state1_batch, 
                                                  time_step=TIME_STEP, 
                                                  hidden_state=h_n,
                                                  cell_state=c_n) ##
        with torch.no_grad():
          Q2, _ = self.model1(bsize=batch_size, input=state2_batch, 
                                                  time_step=TIME_STEP, 
                                                  hidden_state=h_n,
                                                  cell_state=c_n) ##
        Q2_max = torch.max(Q2.detach(),dim=1)[0]
        Y = reward_batch[:, TIME_STEP - 1] + gamma * ((1 - done_batch) * torch.max(Q2.detach(),dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch[:, TIME_STEP - 1].long().unsqueeze(dim=1)).squeeze()

        self.optimizer.zero_grad()
        loss = self.loss_fn(X, Y.detach())
        loss.backward()
        self.optimizer.step()
      return loss

    def updateTargetQ(self):
        self.model2.load_state_dict(self.model1.state_dict())


# In[5]:


# build Double DQN with LSTM neural net models that be be used

class DQN_LSTM(nn.Module):
  def __init__(self, in_size, hid_size, out_size):
    super().__init__()
    self.h_size = hid_size 
    self.in_size = in_size 
    self.out_size = out_size
    self.lstm = nn.LSTM(in_size, hid_size, batch_first=True)
    self.l1 = nn.Linear(hid_size, 4)
    self.relu = nn.ReLU()
  
  def forward(self, bsize, input, time_step, hidden_state, cell_state):
    size_feature = (input.shape[-1])**2 * 3
    input = input.reshape(bsize, time_step, size_feature)
    lstm_out, (h_n, c_n)= self.lstm(input, (hidden_state, cell_state))
    lstm_out = lstm_out[:, time_step - 1, :]
    result = self.l1(lstm_out)
    return result, (h_n, c_n)

  def init_hidden_states(self, bsize):
    h = torch.zeros(1, bsize, self.h_size).float()
    c = torch.zeros(1, bsize, self.h_size).float()
    return h, c

#LSTM model class 
class lstm_modelClass:

    kind = 'modelDQN_LSTM'                    # class variable shared by all instances

    def __init__(self, insize, hidsize, outsize, lr, replaySize):
        self.modeltype = 'lstm'  
        self.model1 = DQN_LSTM(insize, hidsize, outsize)
        self.model2 = DQN_LSTM(insize, hidsize, outsize)
        self.optimizer = torch.optim.Adam(self.model1.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        self.training_tstep = 5  # I assume that we will want to read this in
        self.replay = Memory(replaySize)
        self.epoch_mem = [] 
        self.h_state = torch.zeros(1, 1, hidsize).float() # note manually entered 1 fore bsize
        self.c_state = torch.zeros(1, 1, hidsize).float()
        self.h_size = hidsize
        self.sm = nn.Softmax(dim=1)
        
    def init_hidden_states(self, bsize):
        self.h = torch.zeros(1, bsize, self.h_size).float()
        self.c = torch.zeros(1, bsize, self.h_size).float()
        #return h, c

    def takeAction(self, inputs):
        world, i, j, epsilon, vision = inputs
        img = agentVisualField(world, (i,j), 4)
        input = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
  
        Q, (h_n, c_n) = self.model1(bsize=1, input=input, time_step=1, 
                                           hidden_state=self.h_state,
                                           cell_state=self.c_state) ##
        self.h_state = h_n ##  we should be able to save the current hidden states in the model, no?
        self.c_state = c_n ##  or does this lead to problems elsewhere?
        p = self.sm(Q).detach().numpy()[0]

        if epsilon > .5:
            if (random.random() < epsilon):
                action = np.random.randint(0,len(p))
            else:
                action = np.argmax(Q.detach().numpy())
        else:
            action = np.random.choice(np.arange(len(p)), p = p)
            
        return action, input
    
    def training(self, batch_size, gamma):

      loss = torch.tensor(0.)

      if len(self.replay.memory) > batch_size:  ##
        TIME_STEP = self.training_tstep
        h_n, c_n = self.model1.init_hidden_states(bsize=batch_size) ##

        minibatch = self.replay.get_batch(bsize=batch_size, time_step=TIME_STEP) 
        current_states = []
        acts = []
        rewards = []
        next_states = []
        done_val = []

        for b in minibatch:
            cs, ac, rw, ns, dones = [], [], [], [], []
            for step in b:
                cs.append(step[0])
                ac.append(step[1])
                rw.append(step[2])
                ns.append(step[3])
                dones.append(step[4])
            current_states.append(torch.stack(cs))
            acts.append(ac)
            rewards.append(rw)
            next_states.append(torch.stack(ns))
            done_val.append(dones)

        action_batch = torch.Tensor(acts)
        reward_batch = torch.Tensor(rewards)
        done_batch = torch.Tensor(done_val)
        done_batch = done_batch[:, TIME_STEP - 1]
        state1_batch = torch.stack(current_states)
        state2_batch = torch.stack(next_states)

        Q1, _ = self.model1(bsize=batch_size, input=state1_batch, 
                                                  time_step=TIME_STEP, 
                                                  hidden_state=h_n,
                                                  cell_state=c_n) ##
        with torch.no_grad():
          Q2, _ = self.model1(bsize=batch_size, input=state2_batch, 
                                                  time_step=TIME_STEP, 
                                                  hidden_state=h_n,
                                                  cell_state=c_n) ##
        Q2_max = torch.max(Q2.detach(),dim=1)[0]
        Y = reward_batch[:, TIME_STEP - 1] + gamma * ((1 - done_batch) * torch.max(Q2.detach(),dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch[:, TIME_STEP - 1].long().unsqueeze(dim=1)).squeeze()

        self.optimizer.zero_grad()
        loss = self.loss_fn(X, Y.detach())
        loss.backward()
        self.optimizer.step()
      return loss

    def updateTargetQ(self):
        self.model2.load_state_dict(self.model1.state_dict())

 
  


# In[6]:


# build Double DQN neural net models that be be used

class DQN(nn.Module):

    def __init__(self, numFilters, inSize, outSize, hidsize):
        super(DQN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=numFilters, kernel_size=1)
        self.l2 = nn.Linear(inSize,hidsize)
        self.l3 = nn.Linear(hidsize,hidsize)
        self.l4 = nn.Linear(hidsize,150)
        self.l5 = nn.Linear(150,outSize)
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.conv_bn = nn.BatchNorm2d(5)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        """
        forward of DQN
        """
        x = x/255
        y1 = F.relu(self.conv_layer1(x))
        # ave pool here used to "count" the gems in areas around the agent. Might want to increase stride since should be pretty low res input
        y2 = self.avg_pool(y1)
        y2 = torch.flatten(y2,1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1,y2),1)
        y = F.relu(self.l2(y))
        y = self.dropout(y)
        y = F.relu(self.l3(y))
        #y = self.dropout(y)
        #y = F.relu(self.l4(y))
        #y = self.dropout(y)
        value = self.l5(y)
        return value

class modelClassDQN:

    kind = 'modelDQN'                    # class variable shared by all instances

    def __init__(self, value, lr, replaySize, inSize, outSize, hidsize):
        self.modeltype = 'double_dqn'  
        self.model1 = DQN(value, inSize, outSize, hidsize)
        self.model2 = DQN(value, inSize, outSize, hidsize)
        self.optimizer = torch.optim.Adam(self.model1.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        #self.replay = deque([],maxlen=replaySize)
        self.replay = Memory(replaySize)
        self.epoch_mem = [] 
        self.sm = nn.Softmax(dim=1)
        # below is a hack until the memory works properly
        self.replayDQN = deque([],maxlen=replaySize)

       
    def takeAction(self, inputs):
        world, i, j, vision, epsilon = inputs
        img = agentVisualField(world, (i,j), 4)
        input = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
        Q = self.model1(input)
        p = self.sm(Q).detach().numpy()[0]
        if epsilon > .5:
             if (random.random() < epsilon):
                 action = np.random.randint(0,len(p))
             else:
                 action = np.argmax(Q.detach().numpy())
        if epsilon < .5:
             action = np.random.choice(np.arange(len(p)), p = p)

        return action, input
     
    def training(self, batch_size, gamma):

      loss = torch.tensor(0.)
      if len(self.replayDQN) > batch_size:

        minibatch = random.sample(self.replayDQN, batch_size)
        state1_batch = torch.cat([s1 for (s1,a,r,s2, d) in minibatch]) 
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])      
        
        # note, will need to update the 3,9,9 to use the vision size of the agent

        Q1 = self.model1(state1_batch.reshape(batch_size,3,9,9))
        with torch.no_grad():
          Q2 = self.model2(state2_batch.reshape(batch_size,3,9,9))

        Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2.detach(),dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

        self.optimizer.zero_grad()
        loss = self.loss_fn(X, Y.detach())
        loss.backward()
        self.optimizer.step()
      return loss

    def updateTargetQ(self):
        self.model2.load_state_dict(self.model1.state_dict())





# build Double DQN neural net models that be be used

class multInpDQN(nn.Module):

    def __init__(self, numFilters, inSize, outSize, hidsize):
        super(multInpDQN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=9, out_channels=numFilters, kernel_size=1)
        self.l2 = nn.Linear(inSize,hidsize)
        self.l3 = nn.Linear(hidsize,hidsize)
        self.l4 = nn.Linear(hidsize,150)
        self.l5 = nn.Linear(150,outSize)
        self.avg_pool = nn.MaxPool2d(3, 1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.conv_bn = nn.BatchNorm2d(5)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        """
        forward of DQN
        """
        x = x/255
        y1 = F.relu(self.conv_layer1(x))
        # ave pool here used to "count" the gems in areas around the agent. Might want to increase stride since should be pretty low res input
        y2 = self.avg_pool(y1)
        y2 = torch.flatten(y2,1)
        y1 = torch.flatten(y1, 1)
        y = torch.cat((y1,y2),1)
        y = F.relu(self.l2(y))
        y = self.dropout(y)
        y = F.relu(self.l3(y))
        #y = self.dropout(y)
        #y = F.relu(self.l4(y))
        #y = self.dropout(y)
        value = self.l5(y)
        return value

class modelClassmultInDQN:

    kind = 'modelmultInDQN'                    # class variable shared by all instances

    def __init__(self, value, lr, replaySize, inSize, outSize, hidsize):
        self.modeltype = 'double_dqn'  
        self.model1 = multInpDQN(value, inSize, outSize, hidsize)
        self.model2 = multInpDQN(value, inSize, outSize, hidsize)
        self.optimizer = torch.optim.Adam(self.model1.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        #self.replay = deque([],maxlen=replaySize)
        self.replay = Memory(replaySize)
        self.epoch_mem = [] 
        self.sm = nn.Softmax(dim=1)
        # below is a hack until the memory works properly
        self.replayDQN = deque([],maxlen=replaySize)


        
    def takeAction(self, inputs):
        world, i, j, vision, epsilon = inputs
        img = agentVisualField(world, (i,j), 4)
        input1 = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
        input = torch.cat([input1, world[i,j,0].memories[2], world[i,j,0].memories[1]], dim = 1)
        Q = self.model1(input)
        p = self.sm(Q).detach().numpy()[0]
        if epsilon > .5:
             if (random.random() < epsilon):
                 action = np.random.randint(0,len(p))
             else:
                 action = np.argmax(Q.detach().numpy())
        if epsilon < .5:
             action = np.random.choice(np.arange(len(p)), p = p)

        return action, input
    
 
    
    def training(self, batch_size, gamma):

      loss = torch.tensor(0.)
      if len(self.replayDQN) > batch_size:

        minibatch = random.sample(self.replayDQN, batch_size)
        state1_batch = torch.cat([s1 for (s1,a,r,s2, d) in minibatch]) 
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])      
        
        # note, will need to update the 3,9,9 to use the vision size of the agent

        Q1 = self.model1(state1_batch.reshape(batch_size,9,9,9))
        with torch.no_grad():
          Q2 = self.model2(state2_batch.reshape(batch_size,9,9,9))

        Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2.detach(),dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

        self.optimizer.zero_grad()
        loss = self.loss_fn(X, Y.detach())
        loss.backward()
        self.optimizer.step()
      return loss

    def updateTargetQ(self):
        self.model2.load_state_dict(self.model1.state_dict())



# In[7]:


# create the list of neural net models that can be assigned to agents. right now, just DQN is built

class modelClassPlayer:

    kind = 'humanPlayer'                    # class variable shared by all instances

    def __init__(self, actionSpace, replaySize):
        self.modeltype = 'humanPlayer'  
        self.actionSpace = actionSpace
        self.inputType = 'keyboard'
        self.replay = deque([],maxlen=replaySize)
          
    def takeAction(self, world, i, j, vision, epsilon):
        img = agentVisualField(world, (i,j), 4)
        input = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()

        # note, other subplot can be updated for other information about the game that a model
        # may know - health, points, etc. Current versions do not have, but put in like this
        # to remember generalization of the player display
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()
        
        done = 0
        while done == 0:
            action = int(input("Select Action: "))
            if action in self.actionSpace:
                done = 1
            else:
                print("Please try again. Possible actions are below.")
                print(self.actionSpace)
                # we can have iinputType above also be joystick, or other controller
        
        return action, input
 


# In[8]:


# define the object classes for a game

class Gem:

    kind = 'gem'                    # class variable shared by all instances

    def __init__(self, value, color):
        self.health = 1             # for the gen, whether it has been mined or not
        self.appearence = color    # gems are green
        self.vision = 1             # gems can see one radius around them
        self.policy = "NA"          # gems do not do anything
        self.value = value          # the value of this gem
        self.reward = 0             # how much reward this gem has found (will remain 0)
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 1           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized


class Agent:

    kind = 'agent'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [0.,0.,255.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = model         # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.epoch_mem = []
        self.CurrExp = CurrExp()
        self.memories = deque([],maxlen=3)



        
    def instanceDead(self):
        self.kind = "deadAgent"
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.appearence = [130.,130.,130.] # dead agents are grey
        self.epoch_mem = []
        # note, this has to allow for one last training
        

class deadAgent:

    kind = 'deadAgent'               # class variable shared by all instances

    def __init__(self):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [130.,130.,130.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = "NA"         # agent model here. 
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized
        # self.currentExp = []
        self.CurrExp = ()
        self.epoch_mem = []

class Wolf:

    kind = 'wolf'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [255.,0.,0.]    # agents are red
        self.vision = 4             # agents can see three radius around them
        self.policy = model         # gems do not do anything
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.epoch_mem = []
        self.currExp = CurrExp()
        self.memories = deque([],maxlen=3)


    def instanceDead(self):
        self.kind = "deadwolf"
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.appearence = [130.,130.,130.]    # dead agents are grey
        # note, this has to allow for one last training
        
class deadwolf:

    kind = 'deadwolf'               # class variable shared by all instances

    def __init__(self):
        self.health = 10            # for the agents, this is how hungry they are
        self.appearence = [130.,130.,130.]    # agents are blue
        self.vision = 4             # agents can see three radius around them
        self.policy = "NA"         # agent model here. 
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized


class Wall:

    kind = 'wall'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0            # wall stuff is basically empty
        self.appearence = [153., 51., 102.]    # walls are purple
        self.vision = 0             # wall stuff is basically empty
        self.policy = "NA"          # walls do not do anything
        self.value = 0              # wall stuff is basically empty
        self.reward = -.1             # wall stuff is basically empty
        self.static = 1             # wall stuff is basically empty
        self.passable = 0           # you can't walk through a wall
        self.trainable = 0           # whether there is a network to be optimized

class BlastRay:

    kind = 'blastray'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0            
        self.appearence = [255., 255., 255.]    # blast rays are white
        self.vision = 0             # rays do not see
        self.policy = "NA"          # rays do not think
        self.value = 10              # amount of damage if you are hit by the ray
        self.reward = 0             # rays do not want
        self.static = 1             # rays exist for one turn
        self.passable = 1           # you can't walk through a ray without being blasted
        self.trainable = 0           # rays do not learn
        
        
class EmptyObject:

    kind = 'empty'                  # class variable shared by all instances

    def __init__(self):
        self.health = 0             # empty stuff is basically empty
        self.appearence = [0.,0.,0.]  #empty is well, blank 
        self.vision = 1             # empty stuff is basically empty
        self.policy = "NA"          # empty stuff is basically empty
        self.value = 0              # empty stuff is basically empty
        self.reward = 0             # empty stuff is basically empty
        self.static = 1             # whether the object gets to take actions or not
        self.passable = 1           # whether the object blocks movement
        self.trainable = 0           # whether there is a network to be optimized

class tagAgent:

    kind = 'agent'                  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10            # for the agents, this is how hungry they are
        self.is_it = 0              # everyone starts off not it
        self.appearence = [0., 0., 255.]    # agents are blue when not it
        self.vision = 4             # agents can see three radius around them
        # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.policy = model
        self.value = 0              # agents have no value
        self.reward = 0             # how much reward this agent has collected
        self.static = 0             # whether the object gets to take actions or not
        self.passable = 0           # whether the object blocks movement
        self.trainable = 1           # whether there is a network to be optimized
        self.frozen = 0

    def tag(self):
        if self.is_it == 0:
            self.is_it = 1
            self.appearence = [255, 0., 0.]
            self.frozen = 2
        else:
            self.is_it = 0
            self.appearence = [0., 0., 255]

# In[9]:


# generate the view of an agent. 

def agentVisualField(world, location, k = 4, wall_app = [153., 51., 102.]):
  '''
  Create an agent visual field of size (2k + 1, 2k + 1) pixels
  '''

  bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)
  # instantiate image
  image_r = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
  image_g = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
  image_b = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))

  for i in range(bounds[0], bounds[1] + 1):
    for j in range(bounds[2], bounds[3] + 1):
      # while outside the world array index...
      if i < 0 or j < 0 or i >= world.shape[0] - 1 or j >= world.shape[1]:
        # image has shape bounds[1] - bounds[0], bounds[3] - bounds[2]
        # visual appearance = wall
        image_r[i - bounds[0], j - bounds[2]] = wall_app[0]
        image_g[i - bounds[0], j - bounds[2]] = wall_app[1]
        image_b[i - bounds[0], j - bounds[2]] = wall_app[2]
      else:
        image_r[i - bounds[0], j - bounds[2]] = world[i,j,0].appearence[0]
        image_g[i - bounds[0], j - bounds[2]] = world[i,j,0].appearence[1]
        image_b[i - bounds[0], j - bounds[2]] = world[i,j,0].appearence[2]
  
  image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
  return image


# In[10]:


# create the empty grid world

def createWorld(height, width, layers, defaultObject):
  world = np.full((height, width, layers), defaultObject)
  return world


# In[11]:


# this is bad code below, but as a test of making a visual representation

def createWorldImage(world):
  image_r = np.random.random((world.shape[0],world.shape[0]))
  image_g = np.random.random((world.shape[0],world.shape[0]))
  image_b = np.random.random((world.shape[0],world.shape[0]))

  for i in range(world.shape[0]):
    for j in range(world.shape[0]):
      image_r[i,j] = world[i,j,0].appearence[0]
      image_g[i,j] = world[i,j,0].appearence[1]
      image_b[i,j] = world[i,j,0].appearence[2]

  image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
  return image


# In[12]:


# generate the gem search game objects

agent1 = Agent(0)
agent2 = Agent(1)
gem1 = Gem(5, [0.,255.,0.])
gem2 = Gem(15, [255.,255.,0.])
wolf1 = Wolf(2)
emptyObject = EmptyObject()
walls = Wall()


# In[13]:


# create the instances
def createGemWorld(worldSize, wolfp = 0, gem1p = .115, gem2p = .06):
  gems = 0
  agents = 0
  gems = 0

  # make the world and populate
  world = createWorld(worldSize,worldSize,1,emptyObject)

  for i in range(worldSize):
    for j in range(worldSize):
      obj = np.random.choice([0,1,2,3], p = [wolfp, gem1p, gem2p, 1 - wolfp - gem1p - gem2p])
      if obj == 0:
        world[i,j,0] = wolf1
        agents =+ 1
      if obj == 1:
        world[i,j,0] = gem1
        wolves =+ 1
      if obj == 2:
        world[i,j,0] = gem2
        gems =+ 1

  cBal =  np.random.choice([0,1])
  if cBal == 0:
    world[round(worldSize/2),round(worldSize/2),0] = agent1
    world[round(worldSize/2)+1,round(worldSize/2)-1,0] = agent2
  if cBal == 1:
    world[round(worldSize/2),round(worldSize/2),0] = agent2
    world[round(worldSize/2)+1,round(worldSize/2)-1,0] = agent1

  for i in range(worldSize):
    world[0,i,0] = walls
    world[worldSize-1,i,0] = walls
    world[i,0,0] = walls
    world[i,worldSize-1,0] = walls
    
  return world


# In[14]:


# test the world models

def gameTest(worldSize):
  world = createGemWorld(worldSize)
  image = createWorldImage(world)

  moveList = []
  for i in range(world.shape[0]):
    for j in range(world.shape[0]):
      if world[i,j,0].static == 0:
        moveList.append([i,j])

  img = agentVisualField(world, (moveList[0][0],moveList[0][1]), k = 4)

  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.subplot(1, 2, 2)
  plt.imshow(img)
  plt.show()


# In[15]:


# ---------------------------------------------------------------------
#                      Agent Raygun transition rules (updated)
#                      note, to use standard agent, just have the model 
#                      select actions between 0 and 3
# ---------------------------------------------------------------------

def agentRaygunTransitions(holdObject, action, world, models, i, j, rewards, totalRewards, done, input, expBuff = True):

  newLoc1 = i
  newLoc2 = j

  attLoc1 = i
  attLoc2 = j  
    
  reward = 0

  if action == 0:
    attLoc1 = i-1
    attLoc2 = j

  if action == 1:
    attLoc1 = i+1
    attLoc2 = j

  if action == 2:
    attLoc1 = i
    attLoc2 = j-1

  if action == 3:
    attLoc1 = i
    attLoc2 = j+1

  if world[attLoc1, attLoc2, 0].passable == 1:
    world[i,j,0] = EmptyObject()
    reward = world[attLoc1,attLoc2,0].value
    holdObject.CurrExp.reward =+ reward 
    rewards =+ world[attLoc1,attLoc2,0].value
    world[attLoc1,attLoc2,0] = holdObject
    newLoc1 = attLoc1
    newLoc2 = attLoc2
    totalRewards = totalRewards + rewards
  else:
    if world[attLoc1,attLoc2,0].kind == "wall":
      reward = -.1
      rewards =+ -.1
      # we may also want to kill the agent if it walks into a wolf and give
      # the wolf rewards like the idea below for the dealth of agents


        
  # blasters are not working well, and are clunky written. This can be tightened 

  # if action == 4:
  #   blast = True
  #   attLoc1 = i
  #   counter = 1
  #   while blast == True:
  #       attLoc1 = attLoc1 + 1
  #       counter = counter + 1
  #       if world[attLoc1, attLoc2, 0].kind == "empty":
  #           world[attLoc1, attLoc2, 0] = BlastRay()
  #           reward = -1
  #           holdObject.CurrExp.reward =+ reward
  #       if world[attLoc1, attLoc2, 0].passable == 0:
  #           if world[attLoc1, attLoc2, 0].kind == "wolf":
  #               reward = 0
  #               world[attLoc1, attLoc2, 0].CurrExp = 1
  #               world[attLoc1, attLoc2, 0] = deadwolf()
  #           blast = False
  #       if counter > 3:
  #           blast = False
  #   world[i,j,0] = holdObject
  #
  # if action == 5:
  #   blast = True
  #   attLoc1 = i
  #   counter = 1
  #   while blast == True:
  #       attLoc1 = attLoc1- 1
  #       counter = counter + 1
  #       if world[attLoc1, attLoc2, 0].kind == "empty":
  #           world[attLoc1, attLoc2, 0] = BlastRay()
  #           reward = -1
  #           holdObject.CurrExp =+ reward
  #       if world[attLoc1, attLoc2, 0].passable == 0:
  #           if world[attLoc1, attLoc2, 0].kind == "wolf":
  #               reward = 0
  #               world[attLoc1, attLoc2, 0].CurrExp.done = 1
  #               world[attLoc1, attLoc2, 0] = deadwolf()
  #           blast = False
  #       if counter > 3:
  #           blast = False
  #   world[i,j,0] = holdObject
  #
  # if action == 6:
  #   blast = True
  #   attLoc2 = j
  #   counter = 1
  #   while blast == True:
  #       attLoc2 = attLoc2 + 1
  #       counter = counter + 1
  #       if world[attLoc1, attLoc2, 0].kind == "empty":
  #           world[attLoc1, attLoc2, 0] = BlastRay()
  #           reward = -1
  #           holdObject.CurrExp.reward =+ reward
  #       if world[attLoc1, attLoc2, 0].passable == 0:
  #           if world[attLoc1, attLoc2, 0].kind == "wolf":
  #               reward = 0
  #               world[attLoc1, attLoc2, 0].CurrExp.done = 1
  #               world[attLoc1, attLoc2, 0] = deadwolf()
  #           blast = False
  #       if counter > 3:
  #           blast = False
  #   world[i,j,0] = holdObject
  #
  # if action == 7:
  #   blast = True
  #   attLoc2 = j
  #   counter = 1
  #   while blast == True:
  #       attLoc2 = attLoc2 - 1
  #       counter = counter + 1
  #       if world[attLoc1, attLoc2, 0].kind == "empty":
  #           world[attLoc1, attLoc2, 0] = BlastRay()
  #           reward = -1
  #           holdObject.CurrExp.reward =+ reward
  #       if world[attLoc1, attLoc2, 0].passable == 0:
  #           if world[attLoc1, attLoc2, 0].kind == "wolf":
  #               reward = 0
  #               world[attLoc1, attLoc2, 0].CurrExp.done = 1
  #               world[attLoc1, attLoc2, 0] = deadwolf()
  #           blast = False
  #       if counter > 3:
  #           blast = False
  #   world[i,j,0] = holdObject


        
  if expBuff == True:
     world[newLoc1,newLoc2,0].CurrExp.state = input
     world[newLoc1,newLoc2,0].CurrExp.reward = reward
     world[newLoc1,newLoc2,0].CurrExp.action = action
     if done == 1:
         world[newLoc1,newLoc2,0].CurrExp.done = 1

  return world, models, totalRewards


# In[16]:


# ---------------------------------------------------------------------
#     Wolf transition rules (needs to be checked after all updates)
# ---------------------------------------------------------------------

def wolfTransitions(holdObject, action, world, models, i, j, rewards, wolfEats, done, input, expBuff = True):

  newLoc1 = i
  newLoc2 = j
    
  attLoc1 = i
  attLoc2 = j

  reward = 0

  if action == 0:
    attLoc1 = i-1
    attLoc2 = j

  if action == 1:
    attLoc1 = i+1
    attLoc2 = j

  if action == 2:
    attLoc1 = i
    attLoc2 = j-1

  if action == 3:
    attLoc1 = i
    attLoc2 = j+1
    
  if world[attLoc1, attLoc2, 0].passable == 1:
    world[i,j,0] = EmptyObject()
    reward = (world[attLoc1,attLoc2,0].value/20)
    holdObject.currentExp_reward =+ reward 
    rewards =+ (world[attLoc1,attLoc2,0].value/20)
    world[attLoc1,attLoc2,0] = holdObject
    newLoc1 = attLoc1
    newLoc2 = attLoc2
  else:
    if world[attLoc1,attLoc2,0].kind == "wall":
      reward = -.1
      rewards = rewards - .1
      holdObject.currentExp_reward =+ reward 
    if world[attLoc1,attLoc2,0].kind == "agent":
      reward = 10
      rewards = rewards + 10
      holdObject.currentExp_reward =+ reward 
      wolfEats = wolfEats + 1

      world[attLoc1,attLoc2,0] = deadAgent()

  if expBuff == True:
     world[newLoc1,newLoc2,0].CurrExp.state = input
     world[newLoc1,newLoc2,0].CurrExp.reward = reward
     world[newLoc1,newLoc2,0].CurrExp.action = action
     if done == 1:
         world[newLoc1,newLoc2,0].CurrExp.done = 1
        
  return world, models, wolfEats, newLoc1, newLoc2


# In[17]:


# play and learn the game

def playGame(models, worldSize = 15, epochs = 200000, epsilon = 1, maxEpochs = 100):

  maxEpochs = 100
  losses = 0
  totalRewards = 0
  status = 1
  sync_freq = 500
  game = 0
  turn = 0

  for epoch in range(epochs):
    world =createGemWorld(worldSize)
    for i, j in (findTrainables(world)):
        world[i,j,0].CurrExp = CurrExp()
        world[i,j,0].epoch_mem = []
        img = agentVisualField(world, (i,j), 4)
        initState = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()
        for _ in range(3):
            world[i,j,0].memories.append(initState)
        
        
    
    
    world[i,j,0].done = 0
    
    rewards = 0
    done = 0
    withinTurn = 0
    wolfEats = 0
    agentEats = 0
    
    while(done == 0):
      
      withinTurn = withinTurn + 1
      turn = turn+1
    
      for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i,j,0].kind == "blastray":
                world[i,j,0] = EmptyObject()
    
      if turn % sync_freq == 0: 
        for mods in range(len(models)):
            models[mods].updateTargetQ

      moveList = findMoveables(world)
      random.shuffle(moveList)
        
      if len(moveList) == 0:
        done = 1

      for i, j in moveList:
        holdObject = world[i,j,0]
        
        if holdObject.static != 1:
            action, input = models[holdObject.policy].takeAction([world, i, j, 4, epsilon])
                                                                    
        if withinTurn == maxEpochs:
          done = 1

        if holdObject.kind == "agent":
          # there is way too much being passed in here. Will need to do a careful cleanup of hhe transition functions
          world, models, totalRewards = agentRaygunTransitions(holdObject, action, world, models, i, j, rewards, totalRewards, done, input)

        if holdObject.kind == "wolf":
          world, models, wolfEats, exp = wolfTransitions(holdObject, action, world, models, i, j, rewards, wolfEats, done, input)

      # changes
      useEndExp = True
      if useEndExp == True:
        updateExperiences(world, models, multipleImages = True)
        
    if epsilon > 0.1:
        epsilon -= (1/(turn))

    if epsilon > 0.2:
        if epoch > 1000 and epoch%5000 == 0:
            epsilon -= .05
    
    # code below is not working
    #episilon = updateEpsilon(epoch, epsilon, turn)
    
    # only train at the end of the game, and train each of the models that are in the model list
    # this training is happening accidently at the end of one round. we may want more sparse training
    
    #for i, j in (findTrainables(world)):
    #    models[world[i,j,0].policy].append()
        
    #    replay.add_episode(world[i,j,0].epoch_mem)
    
    for mod in range(len(models)):
      loss = models[mod].training(256, .9)
      losses = losses + loss.detach().numpy()

    if epoch % 100 == 0:
      print(epoch, totalRewards, wolfEats, losses, epsilon)
      wolfEats = 0
      agentEats = 0
      losses = 0
      totalRewards = 0
  return models


# In[18]:


def watchAgame(world, models, epochs, maxEpochs):
  fig = plt.figure()
  ims = []

  totalRewards = 0
  rewards = 0
  turn = 0
  wolfEats = 0
  epsilon = 0.2

  for epoch in range(epochs):
    
    done = 0
    withinTurn = 0
    while(done == 0):
        
      image = createWorldImage(world)
      im = plt.imshow(image, animated=True)
      ims.append([im])
    
      withinTurn = withinTurn + 1
      turn = turn+1
    
      for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if world[i,j,0].kind == "blastray":
                world[i,j,0] = EmptyObject()
      
      moveList = findMoveables(world)
      random.shuffle(moveList)

      for i, j in moveList:
        holdObject = world[i,j,0]
        
        img = agentVisualField(world, (i,j), holdObject.vision)
        input = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).float()

        if holdObject.static != 1:
            action, input = models[holdObject.policy].takeAction([world, i, j, 4, epsilon])
        
        if holdObject.kind == "agent":
          world, models, totalRewards = agentRaygunTransitions(holdObject, action, world, models, i, j, rewards, totalRewards, done, input)

        if holdObject.kind == "wolf":
          world, models, wolfEats = wolfTransitions(holdObject, action, world, models, i, j, rewards, wolfEats, done, input)


        
        if withinTurn == maxEpochs:
          done = 1
        
  ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
  return ani


# In[19]:


# view a replay memory

def examineReplay(models, index, modelnum):

  image_r =  models[modelnum].replay[index][0][0][0].numpy()
  image_g =  models[modelnum].replay[index][0][0][1].numpy()
  image_b =  models[modelnum].replay[index][0][0][2].numpy()

  image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

  image_r =  models[modelnum].replay[index][3][0][0].numpy()
  image_g =  models[modelnum].replay[index][3][0][1].numpy()
  image_b =  models[modelnum].replay[index][3][0][2].numpy()

  image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)  

  action = models[modelnum].replay[index][1]
  reward = models[modelnum].replay[index][2]
  done = models[modelnum].replay[index][4]

  return image, image2, (action, reward, done)


# In[20]:


# look at a few replay games

def replayGames(numGames, modelNum, startingMem):
  for i in range(numGames):
    image, image2, memInfo = examineReplay(models, i+startingMem,modelNum)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="Blues_r")
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap="Accent_r")

    print(memInfo)

    plt.show()


def examineReplayMemory(models, episode, index, modelnum):

  image_r =  models[modelnum].replay.memory[episode][index][0][0][0].numpy()
  image_g =  models[modelnum].replay.memory[episode][index][0][0][1].numpy()
  image_b =  models[modelnum].replay.memory[episode][index][0][0][2].numpy()

  image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

  image_r =  models[modelnum].replay.memory[episode][index][3][0][0].numpy()
  image_g =  models[modelnum].replay.memory[episode][index][3][0][1].numpy()
  image_b =  models[modelnum].replay.memory[episode][index][3][0][2].numpy()

  image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)  

  action = models[modelnum].replay.memory[episode][index][1]
  reward = models[modelnum].replay.memory[episode][index][2]
  done = models[modelnum].replay.memory[episode][index][4]

  return image, image2, (action, reward, done)


# In[20]:


# look at a few replay games

def replayMemoryGames(models, modelNum, episode):
    epLength = len(models[modelNum].replay.memory[episode])
    for i in range(epLength):
        image, image2, memInfo = examineReplayMemory(models, episode, i,modelNum)
    
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")
    
        print(memInfo)
    
        plt.show()


def numberOfMemories(modelNum, models):
    episodes = len(models[modelNum].replay.memory)
    print("there are ", episodes, " in the model replay buffer.")
    for e in range(episodes):
        epLength = len(models[0].replay.memory[e])
        print("Memory ", e, " is ", epLength, " long.")
    


# In[21]:


# check visual layout

# gameTest(25)


# In[ ]:


loadModels = 0
if loadModels == 1:
    print("Loading Models")
    with open("modelFile_A2", "rb") as fp:
        models = pickle.load(fp)
    world = createGemWorld(30)
    ani1 = watchAgame(world, models, 1, 200)
    ani1.save("animation_A2rand-30.gif", writer='pillow', fps=8)

newModels = 1

# create neuralnet models
if newModels == 1:
    print("Generating Models")
    models = []
    models.append(modelClassmultInDQN(5,.0001, 1000, 650, 4, 150)) # agent1 model
    models.append(modelClassmultInDQN(5,.0001, 1000, 650, 4, 150))  # agent2 model
    #models.append(lstm_cnn_modelClass(5,.0001, 1000, 650, 4, 400))  # wolf model
    
  
    print("Testing Models")
    models = playGame(models, worldSize=15, epochs=50000, epsilon=.9, maxEpochs = 25)
    with open("modelFile_A4", "wb") as fp:
        pickle.dump(models, fp)
    world =createGemWorld(30)
    ani1 = watchAgame(world, models, 1, 100)
    ani1.save("animation_A4.gif", writer='pillow', fps=6)
    

    
newModels = 1
if newModels == 2:


    print("Testing Models")
    models = playGame(models, 15, 10000, .8, maxEpochs = 25)
    with open("modelFile_B", "wb") as fp:
        pickle.dump(models, fp)
    world =createGemWorld(30)
    ani1 = watchAgame(world, models, 1, 50)
    ani1.save("animation_B.gif", writer='PillowWriter', fps=2)
    
    print("Testing Models")
    models = playGame(models, 15, 10000, .3, maxEpochs = 25)
    with open("modelFile_sm1", "wb") as fp:
        pickle.dump(models, fp)
    world =createGemWorld(30)
    ani1 = watchAgame(world, models, 1, 50)
    ani1.save("animation_sm1.gif", writer='PillowWriter', fps=2)
    
    print("Testing Models")
    models = playGame(models, 15, 10000, .3, maxEpochs = 25)
    with open("modelFile_C", "wb") as fp:
        pickle.dump(models, fp)
    world =createGemWorld(30)
    ani1 = watchAgame(world, models, 1, 50)
    ani1.save("animation_C.gif", writer='PillowWriter', fps=2)
    
    print("Testing Models")
    models = playGame(models, 25, 10000, .3, maxEpochs = 100)
    with open("modelFile_sm2", "wb") as fp:
        pickle.dump(models, fp)
    world =createGemWorld(100)
    ani1 = watchAgame(world, models, 1, 50)
    ani1.save("animation_sm2.gif", writer='PillowWriter', fps=2)
    

newModels = 1
if newModels == 3:
        for games in range(20):
            models = playGame(models, 20, 10000, .3, maxEpochs = 40)
            world =createGemWorld(30)
            ani1 = watchAgame(world, models, 1, 100)
            ani1.save("animation_r"+str(games)+".gif", writer='PillowWriter', fps=2)
            with open("modelFile_r"+str(games), "wb") as fp:
                pickle.dump(models, fp)

#
# # In[ ]:
#
#
# humanPlayer = 1
# if humanPlayer == 1:
#     with open("modelFile_17", "rb") as fp:
#         models = pickle.load(fp)
#     models[1] = modelClassPlayer([0,1,2,3,4,5,6,7], 10)
#     models = playGame(models, 25, 10000, .3)
#
#
# # In[ ]:
#
#
# # watch game
# # (these are not ideal videos and need to be updated. Not sure why the files are movies in) jupyper but not as files)
#
# world =createGemWorld(30)
# ani1 = watchAgame(world, models, 1, 100)
# ani1.save('animation.gif', writer='PillowWriter', fps=2)
#
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# # examine replay
# # numGames, modelNum, startingMem)
#
# replayGames(100, 2, 100)
#
#
# # In[ ]:
#
#
# # the wolf model is from wolves chasing random agents
#
# with open("wolfmodel", "rb") as fp:
#     wolfmodel = pickle.load(fp)
#
# # the agent model is from below
#
# with open("modelFile_3", "rb") as fp:
#     models = pickle.load(fp)
#
# # the logic here is that the agents are learning to escape the wolves faster than they can learn to eat them
# # so evening the playing field a little
#
# # place the wolf model into the main model
# models[2] = wolfmodel[2]
#
# for games in range(4):
#     models = playGame(models, 25, 10000, .3)
#     world =createGemWorld(30)
#     ani1 = watchAgame(world, models, 1, 100)
#     ani1.save("animation_comb"+str(games)+".gif", writer='PillowWriter', fps=2)
#     with open("modelFile_comb_"+str(games), "wb") as fp:
#         pickle.dump(models, fp)
#
#
#
#
#
# # In[ ]:
#
#
# print(len(models[0].replay.memory))
#
#
# # In[ ]:
#
#
# # notes
# # models need to have inputs for image and actions
# # action selection needs to select as a function of action space
#
# # function to auto-compute the vision input from vision parameters
#
# # need to be able to read other models
# # separate training games (train the wolves and the agents separately for a time, etc.)
#
# # model needs to be able to automatically set up the right model, and create the right variables
# # previous experiences for LTSM likely need to be in the agent rather than the model so that other agents can
# # share the policy model and replays, but not the direct experience of the agent
#
# # maybe want a separate select action function to clean up a little bit (reads in epsiolon, etc)
# # should the game transition files be one file, where it reads in instance class type?
#
# # if the first part of e-greedy learning is basically random actions, why not generate a MASSIVE amount of
# # game data, save it to disk, and then have the neural net learn from the data from the first round to get the
# # intial Q models. This can be random examples, or even carefully planned ones (wolf attack is a small world that
# # when the wolf finally eats the agent - the agent can be slower). Eating, where you have two turns to get the
# # bush that is right next to you (initial wolf attack can be like that too, can even be one turn and we make
# # the rest of the background random, so the important thing to learn is how to eat before finding our how to
# # hunt or forage)
#
# visionSize = 4
# filters = 5
# avePoolSize = 3
# conv11out = ((visionSize*visionSize)+1)*filters
# avePoolSizeOut = (visionSize-2)*(visionSize-2)*filters
#
# # will the LSTM be failing if different agents are sharing a memory system?

