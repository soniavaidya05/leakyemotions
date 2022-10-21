from collections import deque
import random
#from models.linear_lstm_dqn_PER import Model_linear_LSTM_DQN
import numpy as np
import torch

save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

device = "cpu"
print(device)


from astropy.visualization import make_lupton_rgb



class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def _number_memories(self):
        return self.n_entries

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(
        self, capacity, e=0.01, a=0.06, beta=0.4, beta_increment_per_sampling=0.0001
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def number_memories(self):
        return self.tree._number_memories()

from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize

# from models.memory import Memory
#from models.perception import agent_visualfield


import random
import numpy as np
from collections import deque

#from models.priority_replay import Memory, SumTree


class LSTM_DQN(nn.Module):
    """
    TODO: need to be able to have an input for non CNN layers to add additional inputs to the model
            likely requires an MLP before the LSTM where the CNN and the additional
            inputs are concatenated
    """

    def __init__(
        self,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        n_layers=1,
        batch_first=True,
    ):
        super(LSTM_DQN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=hid_size1,
            num_layers=n_layers,
            batch_first=True,
        )
        self.l1 = nn.Linear(hid_size1, hid_size1)
        self.l2 = nn.Linear(hid_size1, hid_size2)
        self.l3 = nn.Linear(hid_size2, out_size)
        self.dropout = nn.Dropout(0.15)
        

    def forward(self, x):
        """
        TODO: check the shapes below. 
        """
        #r_in =normalize(x, p=1.0, dim = 1)
        r_in = x

        #print("r_in.shape: ", r_in.shape)
        r_out, (h_n, h_c) = self.rnn(r_in)
        #print("r_out.shape: ", r_out.shape)
        y = F.relu(self.l1(r_out[:, -1, :]))
        # y = F.relu(self.l1(r_out))
        #print("y.shape: ", y.shape)
        y = F.relu(self.l2(y))
        y = self.l3(y)

        return y


class Model_linear_LSTM_DQN:

    kind = "linear_lstm_dqn"  # class variable shared by all instances

    def __init__(
        self,
        lr,
        replay_size,
        in_size,
        hid_size1,
        hid_size2,
        out_size,
        priority_replay=True,
        device="cpu",
    ):
        self.modeltype = "linear_lstm_dqn"
        self.model1 = LSTM_DQN(
            in_size, hid_size1, hid_size2, out_size
        )
        self.model2 = LSTM_DQN(
            in_size, hid_size1, hid_size2, out_size
        )
        self.optimizer = torch.optim.Adam(
            self.model1.parameters(), lr=lr, weight_decay=0.01
        )
        self.loss_fn = nn.MSELoss()
        self.sm = nn.Softmax(dim=1)
        self.priority_replay = priority_replay
        if priority_replay == True:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0.6,  # set this to 0 for uniform sampling (check these numbers)
                beta=0.4,  # 0.4, set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0.0001,  # set this to 0 for uniform sampling (check these numbers)
            )
        if priority_replay == False:
            self.max_priority = 1.0
            self.PER_replay = Memory(
                replay_size,
                e=0.01,
                a=0,  # set this to 0 for uniform sampling (check these numbers)
                beta=0,  # set this to 0 for uniform sampling (check these numbers)
                beta_increment_per_sampling=0,  # set this to 0 for uniform sampling (check these numbers)
            )
        self.device = device

    def pov(self, world, location, holdObject, inventory=[]):
        """
        TODO: rewrite pov to simply take in a vector
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :] = previous_state[:, 1:, :]

        #state_now = torch.tensor(inventory).float()
        #current_state[:, -1, :] = state_now

        return current_state

    def take_action(self, params):
        """
        Takes action from the input
        """

        inp, epsilon = params

        Q = self.model1(inp)
        p = self.sm(Q).cpu().detach().numpy()[0]

        use_softmax = False
        if use_softmax == False:
            epsilon = max(epsilon, .1)
            if random.random() < epsilon:
                action = np.random.randint(0, len(p))
            else:
                action = np.argmax(Q.detach().cpu().numpy())

        if use_softmax == True: # AIecon simple fails with this
            if epsilon > 0.3:
                if random.random() < epsilon:
                    action = np.random.randint(0, len(p))
                else:
                    action = np.argmax(Q.detach().cpu().numpy())
            else:
                action = np.random.choice(np.arange(len(p)), p=p)

        return action

    def training(self, batch_size, gamma):
        """
        DQN batch learning
        """
        loss = torch.tensor(0.0)

        current_replay_size = batch_size + 1

        if current_replay_size > batch_size:

            # note, rewrite to be a min of batch_size or current_replay_size
            # need to figure out how to get current_replay_size
            minibatch, idxs, is_weight = self.PER_replay.sample(batch_size)

            # the do(device) below should not be necessary
            # but on mps, action, reward, and done are being bounced back to the cpu
            # currently removed for a test on CUDA

            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch]).to(self.device)
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch]).to(self.device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch]).to(self.device)

            Q1 = self.model1(state1_batch)
            with torch.no_grad():
                Q2 = self.model2(state2_batch)

            Y = reward_batch + gamma * (
                (1 - done_batch) * torch.max(Q2.detach(), dim=1)[0]
            )

            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            errors = torch.abs(Y - X).data.cpu().numpy()

            # there should be better ways of doing the following
            self.max_priority = np.max(errors)

            # update priority
            for i in range(len(errors)):
                idx = idxs[i]
                self.PER_replay.update(idx, errors[i])

            self.optimizer.zero_grad()
            if self.priority_replay == False:
                loss = self.loss_fn(X, Y.detach())
            if self.priority_replay == True:
                replay_stable = 0
                if replay_stable == 1:
                    loss = self.loss_fn(X, Y.detach())
                if replay_stable == 0:
                    # loss = (
                    #    torch.FloatTensor(is_weight).to(self.device) * F.mse_loss(Y, X)
                    # ).mean()
                    # compute this twice!
                    loss = (
                        torch.FloatTensor(is_weight).to(self.device)
                        * ((X - Y.detach()) ** 2)
                    ).mean()
            # the step below is where the M1 chip fails
            loss.backward()
            self.optimizer.step()
        return loss

    def updateQ(self):
        """
        Update double DQN model
        """
        self.model2.load_state_dict(self.model1.state_dict())

    def transfer_memories(self, world, loc, extra_reward=True, seqLength=4):
        """
        Transfer the indiviu=dual memories to the model
        TODO: We need to have a single version that works for both DQN and
              Actor-criric models (or other types as well)
        """
        exp = world[loc].episode_memory[-1]
        high_reward = exp[1][2]

        # move experience to the gpu if available
        exp = (
            exp[0],
            (
                exp[1][0].to(self.device),
                torch.tensor(exp[1][1]).float().to(self.device),
                torch.tensor(exp[1][2]).float().to(self.device),
                exp[1][3].to(self.device),
                torch.tensor(exp[1][4]).float().to(self.device),
            ),
        )

        self.PER_replay.add(exp[0], exp[1])
        if extra_reward == True and abs(high_reward) > 9:
            for _ in range(seqLength):
                self.PER_replay.add(exp[0], exp[1])








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

    def init_replay(self, numberMemories):
        image = torch.zeros(1, numberMemories, 6).float()
        exp = (0.1, (image, 0, 0, image, 0))
        self.episode_memory.append(exp)


    def transition(self, env, models, action, done, location):
        new_loc = location
        reward = 0

        if action == 0:
            if random.random() < self.wood_skill:
                self.wood = max(self.wood + 1,3)
                reward = 0.00 # for this to really be working, this needs to be zero
        if action == 1:
            if random.random() < self.stone_skill:
                self.stone = max(self.stone + 1,3)
                reward = 0.00 # for this to really be working, this needs to be zero
        if action == 2:
            dice_role = random.random()
            if dice_role < self.house_skill and self.wood > 0 and self.stone > 0:
                self.wood = self.wood - 1
                self.stone = self.stone - 1
                self.house = self.house + 1
                self.coin = self.coin + 10
                reward = 10
                #print(self.policy, " built a house! Rolled a ", dice_role)
        if action == 3:
            if self.wood > 1:
                self.wood = self.wood - 2
                reward = 1
                self.coin = self.coin + 1
                env.wood = env.wood + 1
            #if self.wood < 2:
            #    reward = -.1
        if action == 4:
            if self.stone > 1:
                self.stone = self.stone - 2
                reward = 1
                self.coin = self.coin + 1
                env.stone = env.stone + 1
            #if self.stone < 2:
            #    reward = -.1
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

        next_state = generate_input(agent_list, agent).unsqueeze(0).to(device)

        #next_state = torch.tensor([self.wood, self.stone, self.coin]).float().unsqueeze(0)

        return env, reward, next_state, done, new_loc


class AIEcon_simple_game:
    def __init__(self):
        self.wood = 6
        self.stone = 6

    def pov(self, world, location, holdObject, inventory=[]):
        """
        TODO: rewrite pov to simply take in a vector
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :] = previous_state[:, 1:, :]


        current_state[:, -1, :] = state

        return current_state


def create_models():
    models = []
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.0005,
            replay_size=1024,  
            in_size=6,  
            hid_size1=10,  
            hid_size2=10,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model1
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.0005,
            replay_size=1024,  
            in_size=6,  
            hid_size1=10,  
            hid_size2=10,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model2
    models.append(
        Model_linear_LSTM_DQN(
            lr=0.0005,
            replay_size=1024,  
            in_size=6,  
            hid_size1=10,  
            hid_size2=10,  
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
num_agents = 18
for i in range(num_agents):
    agent_type = np.random.choice([0,1,2])
    if agent_type == 0:
        appearence = [1, 0, 0, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
        agent_list.append(Agent(0,0,appearence, .95, .25, .1))
    if agent_type == 1:
        appearence = [0, 1, 0, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
        agent_list.append(Agent(1,1,appearence, .25, .95, .1))
    if agent_type == 2:
        appearence = [0, 0, 1, np.random.choice([0,1]), np.random.choice([0,1]), np.random.choice([0,1])]
        agent_list.append(Agent(2,2,appearence, .1, .1, .95))
    agent_list[i].init_replay(3)   

rewards = [0,0,0]
losses = 0
model_learn_rate = 2
sync_freq = 500

trainable_models = [0,1,2]
agent1_actions = [0,0,0,0,0,0,0]
agent2_actions = [0,0,0,0,0,0,0]
agent3_actions = [0,0,0,0,0,0,0]

epsilon = .99

def generate_input(agent_list, agent):
    # need to set this up for LSTM input
    #cur_wood = np.log(agent_list[agent].wood+1)
    #cur_stone = np.log(agent_list[agent].stone+1)
    #cur_coin = np.log(agent_list[agent].coin+1)

    cur_wood = agent_list[agent].wood /5 # what is the best way of having current inventory
    cur_stone = agent_list[agent].stone /5
    cur_coin = agent_list[agent].coin /5

    suf_wood = 0    
    suf_stone = 0
    suf_coin = 0
    if agent_list[agent].wood > 2: # does it make sense to also have a degree of whether it can buy or sell as a binary?
        suf_wood = 1
    else:
        suf_wood = 0
    if agent_list[agent].stone > 2:
        suf_stone = 1
    else:
        suf_stone = 0
    if agent_list[agent].coin > 2:
        suf_coin = 1
    else:
        suf_coin = 0
    state = torch.tensor([cur_wood, cur_stone, cur_coin, suf_wood, suf_stone, suf_coin]).float()

    return state

def prepare_lstm(agent_list, agent, state):
    previous_state = agent_list[agent].episode_memory[-1][1][0]
    current_state = previous_state.clone()
    current_state[:, 0:-1, :] = previous_state[:, 1:, :]
    current_state[:, -1, :] = state
    return current_state

def prepare_lstm2(previous_state, next_state):

    current_state = previous_state.clone()
    current_state[:, 0:-1, :] = previous_state[:, 1:, :]
    current_state[:, -1, :] = next_state

    return current_state


max_turns = 50

for epoch in range(1000000):
    done = 0
    if epoch % round(sync_freq/max_turns) == 0:
            # update the double DQN model ever sync_frew
            for mods in trainable_models:
                models[mods].model2.load_state_dict(
                    models[mods].model1.state_dict()
                )
    if epoch % round(1000/max_turns) == 0:
        epsilon = epsilon - .001

    env.wood = 10
    env.stone = 10

    for agent in range(len(agent_list)):
        agent_list[agent].coin = 0
        agent_list[agent].wood = 0
        agent_list[agent].stone = 0
        if agent_list[agent].policy == 2:
            agent_list[agent].coin = 6
        agent_list[i].init_replay(3)

    turn = 0
    while done != 1:
        turn = turn + 1
        if turn > max_turns:
            done = 1
        
        for agent in range(len(agent_list)):

            cur_wood = agent_list[agent].wood
            cur_stone = agent_list[agent].stone
            cur_coin = agent_list[agent].coin

            state = generate_input(agent_list, agent).unsqueeze(0).to(device)
            state_lstm = prepare_lstm(agent_list, agent, state)
            action = models[agent_list[agent].policy].take_action([state_lstm, epsilon])
            #print(action)
            env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [])
            rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward
            if agent_list[agent].policy == 0:
                agent1_actions[action] = agent1_actions[action] + 1
            if agent_list[agent].policy == 1:
                agent2_actions[action] = agent2_actions[action] + 1
            if agent_list[agent].policy == 2:
                agent3_actions[action] = agent3_actions[action] + 1
            # is this next state a problem in the main version of gem?
            next_state_lstm = prepare_lstm2(state_lstm, next_state)

            exp = [1, (
                state_lstm,
                action,
                reward,
                next_state_lstm,
                done,
            )]
            
            agent_list[agent].episode_memory.append(exp)

            models[agent_list[agent].policy].PER_replay.add(exp[0], exp[1])

        if turn % model_learn_rate == 0:
            for mods in trainable_models:
                loss = models[mods].training(128, .9) # reducing gamma to see if future Q is the problem
                losses = losses + loss.detach().cpu().numpy()

    if turn % model_learn_rate == 0:
        for mods in trainable_models:
            loss = models[mods].training(128, .9) # reducing gamma to see if future Q is the problem
            losses = losses + loss.detach().cpu().numpy()

    if epoch % round(500/max_turns) == 0:
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


