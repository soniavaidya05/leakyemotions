from collections import deque
import random
from models.linear_lstm_dqn_PER import Model_linear_LSTM_DQN
import numpy as np
import torch

save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

device = "cpu"
print(device)

class Agent():
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # init agents
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.episode_memory = deque([], maxlen=3)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"
        self.wood = 2
        self.stone = 2
        self.house = 0
        self.wood_skill = 0
        self.stone_skill = 0
        self.house_skill = 0
        self.coin = 0
        self.agent_type = 3
        self.create_identity()

    def create_identity(self):
        self.agent_type = np.random.choice([0,1,2])
        if self.agent_type == 0:
            self.appearence = [1, 0, 0, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            self.wood_skill = .9
            self.stone_skill = 0
            self.house_skill = 0
            self.policy = self.agent_type
        if self.agent_type == 1:
            self.appearence = [0, 1, 0, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            self.wood_skill = 0
            self.stone_skill = .9
            self.house_skill = 0
            self.policy = self.agent_type
        if self.agent_type == 2:
            self.appearence = [0, 0, 1, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
            self.wood_skill = 0
            self.stone_skill = 0
            self.house_skill = .9
            self.policy = self.agent_type

    def transition(self, env, models, action, location):
        new_loc = location
        reward = 0
        done = 0

        if action == 0:
            if random.random() < self.wood_skill:
                self.wood = self.wood + 1
                reward = .1
        if action == 1:
            if random.random() < self.stone_skill:
                self.stone = self.stone + 1
                reward = .1
        if action == 2:
            dice_role = random.random()
            if dice_role < self.house_skill and self.wood > 0 and self.stone > 0:
                self.wood = self.wood - 1
                self.stone = self.stone - 1
                self.house = self.house + 1
                self.coin = self.coin + 10
                reward = 25
                #print(self.policy, " built a house! Rolled a ", dice_role)
        if action == 3:
            if self.wood > 2:
                self.wood = self.wood - 2
                reward = 1
                self.coin = self.coin + 1
                env.wood = env.wood + 1
        if action == 4:
            if self.stone > 2:
                self.stone = self.stone - 2
                reward = 1
                self.coin = self.coin + 1
                env.stone = env.stone + 1
        if action == 5:
            if env.wood > 2 and self.coin > 1:
                env.wood = env.wood - 2
                reward = -1
                self.coin = self.coin - 1
                self.wood = self.wood + 2
        if action == 6:
            if env.stone > 2 and self.coin > 1:
                env.stone = env.stone - 2
                reward = -1
                self.coin = self.coin - 1
                self.stone = self.stone + 2

        next_state = torch.tensor([self.wood, self.stone, self.coin]).float().unsqueeze(0)

        return env, reward, next_state, done, new_loc


class AIEcon_simple_game:
    def __init__(self):
        self.wood = 0
        self.stone = 0

def create_models():
    models = []
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.001,
            replay_size=1024,  
            in_size=3,  
            hid_size1=30,  
            hid_size2=30,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model1
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.001,
            replay_size=1024,  
            in_size=3,  
            hid_size1=30,  
            hid_size2=30,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model2
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.001,
            replay_size=1024,  
            in_size=3,  
            hid_size1=30,  
            hid_size2=30,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model3

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models

# AI_econ test game


models = create_models()

env = AIEcon_simple_game()

agent_list = []
num_agents = 30
for i in range(num_agents):
    agent_list.append(Agent(0))
    print(agent_list[i].policy)

rewards = [0,0,0]
losses = 0
model_learn_rate = 2
sync_freq = 500

trainable_models = [0,1,2]
agent1_actions = [0,0,0,0,0,0,0]
agent2_actions = [0,0,0,0,0,0,0]
agent3_actions = [0,0,0,0,0,0,0]

epsilon = .99

for epoch in range(1000000):
    #print("epoch: ", epoch, "env stone and wood: ", env.stone, env.wood)
    if epoch % sync_freq == 0:
            # update the double DQN model ever sync_frew
            for mods in trainable_models:
                models[mods].model2.load_state_dict(
                    models[mods].model1.state_dict()
                )
    if epoch % 1000 == 0:
        epsilon = epsilon - .01

    if epoch % 30 == 0:
        env.wood = 0
        env.stone = 0

    for agent in range(len(agent_list)):
        if epoch % 30 == 0:
            agent_list[agent].coin = 0
            agent_list[agent].wood = 2
            agent_list[agent].stone = 2
        state = torch.tensor([agent_list[agent].wood, agent_list[agent].stone, agent_list[agent].coin]).float().unsqueeze(0).to(device)
        action = models[agent_list[agent].policy].take_action([state, epsilon])
        env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, [])
        rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward
        if agent_list[agent].policy == 0:
            agent1_actions[action] = agent1_actions[action] + 1
        if agent_list[agent].policy == 1:
            agent2_actions[action] = agent2_actions[action] + 1
        if agent_list[agent].policy == 2:
            agent3_actions[action] = agent3_actions[action] + 1

        exp = [1, (
            state,
            action,
            reward,
            next_state,
            done,
        )]
        
        agent_list[agent].episode_memory.append(exp)

        models[agent_list[agent].policy].PER_replay.add(exp[0], exp[1])

    if epoch % model_learn_rate == 0:
        for mods in trainable_models:
            loss = models[mods].training(100, .1) # reducing gamma to see if future Q is the problem
            losses = losses + loss.detach().cpu().numpy()

    if epoch % 500 == 0:
        print("--------------------------------------")
        print("epoch:" , epoch, "loss: ",losses, "points (wood, stone, house): ", rewards, "epsilon: ", epsilon)
        print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
        print("agent1 behaviours: ", agent1_actions)
        print("agent2 behaviours: ", agent2_actions)
        print("agent3 behaviours: ", agent3_actions)
        rewards = [0,0,0]
        losses = 0
        agent1_actions = [0,0,0,0,0,0,0]
        agent2_actions = [0,0,0,0,0,0,0]
        agent3_actions = [0,0,0,0,0,0,0]



agent = 0

for epoch in range(10):
    state = torch.tensor([agent_list[agent].wood, agent_list[agent].stone, agent_list[agent].coin]).float().unsqueeze(0).to(device)
    print(int(state[0][0].numpy()), int(state[0][1].numpy()), int(state[0][2].numpy()))
    action = models[agent_list[agent].policy].take_action([state, epsilon])
    env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, [])

    print(action, reward)


