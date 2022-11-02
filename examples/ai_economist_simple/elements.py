import torch
import random
from collections import deque
from examples.ai_economist_simple.env import generate_input, prepare_lstm, prepare_lstm2


class Agent():
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model, agent_type, appearance, wood_skill, stone_skill, house_skill):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = appearance  # init agents
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.episode_memory = deque([], maxlen=3)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"
        self.wood = 0
        self.stone = 0
        self.house = 0
        self.wood_skill = wood_skill
        self.stone_skill = stone_skill
        self.house_skill = house_skill
        self.coin = 6
        self.agent_type = agent_type
        self.init_rnn_state = None

    def init_replay(self, numberMemories):
        image = torch.zeros(1, numberMemories, 6).float()
        exp = (0.1, (image, 0, 0, image, 0, None))
        self.episode_memory.append(exp)


    def transition(self, env, models, action, done, location, agent_list, agent):
        new_loc = location
        reward = 0

        if action == 0:
            if random.random() < self.wood_skill:
                self.wood = min(self.wood + 1,3)
                reward = 0.00 # for this to really be working, this needs to be zero
        if action == 1:
            if random.random() < self.stone_skill:
                self.stone = min(self.stone + 1,3)
                reward = 0.00 # for this to really be working, this needs to be zero
        if action == 2:
            dice_role = random.random()
            if dice_role < self.house_skill and self.wood > 0 and self.stone > 0:
                self.wood = self.wood - 1
                self.stone = self.stone - 1
                self.house = self.house + 1
                self.coin = self.coin + 10
                reward = 10
        if action == 3:
            #if random.random() < self.wood_skill: # simulates the AI market
            if self.wood > 1:
                self.wood = self.wood - 2
                reward = 1
                self.coin = self.coin + 1
                env.wood = env.wood + 1
        if action == 4:
            #if random.random() < self.stone_skill: # simulates the AI market
            if self.stone > 1:
                self.stone = self.stone - 2
                reward = 1
                self.coin = self.coin + 1
                env.stone = env.stone + 1
        if action == 5:
            if env.wood > 2 and self.coin > 1:
                env.wood = env.wood - 2
                reward = -1.25
                self.coin = self.coin - 1
                self.wood = self.wood + 2
        if action == 6:
            if env.stone > 2 and self.coin > 1:
                env.stone = env.stone - 2
                reward = -1.25
                self.coin = self.coin - 1
                self.stone = self.stone + 2

        next_state = generate_input(agent_list, agent).unsqueeze(0).to(models[0].device)

        #next_state = torch.tensor([self.wood, self.stone, self.coin]).float().unsqueeze(0)

        return env, reward, next_state, done, new_loc
