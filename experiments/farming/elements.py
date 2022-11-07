import torch
import random
from collections import deque
from experiments.ai_economist_simple.env import generate_input, prepare_lstm, prepare_lstm2


class Farmer():
    kind = "farmer"  # class variable shared by all instances

    def __init__(self, model, agent_type, appearance, wood_skill, stone_skill, house_skill):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = appearance  # init agents
        self.policy = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.reward = 0  # how much reward this agent has collected
        self.episode_memory = deque([], maxlen=3)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"
        self.agent_type = agent_type
        self.init_rnn_state = None
        self.state = torch.zeros(6).float()
        self.loc = 0

    def init_replay(self, numberMemories):
        image = torch.zeros(1, numberMemories, 6).float()
        exp = (0.1, (image, 0, 0, image, 0, None))
        self.episode_memory.append(exp)


    def transition(self, env, models, action, done, location, agent_list, agent):
        new_loc = location
        reward = 0

        if action == 0:
            if location == 0:
                env.world[0,1,0].agentList.append(self)
                env.world[0,0,0].agentList.pop(self)
                self.loc = (0,1,0)
            if location == 1:
                env.world[0,0,0].agentList.append(self)
                env.world[0,1,0].agentList.pop(self)
                self.loc = (0,0,0)
            reward = 0

        if action == 1:
            reward = env.world[location].numAgents /2

        # get this below (need to figure out the action space)
        next_state, _ = generate_input(agent_list, agent, agent_list[agent].state)
        next_state = next_state.unsqueeze(0).to(models[0].device)


        return env, reward, next_state, done, new_loc

class Farm():
    kind = "farmer"  # class variable shared by all instances

    def __init__(self, model, agent_type, appearance, wood_skill, stone_skill, house_skill):
        self.appearance = None  # init agents
        self.episode_memory = deque([], maxlen=3)  # we should read in these maxlens
        self.has_transitions = False
        self.action_type = "static"
        self.agentList = []
        self.numAgents = 0

 

