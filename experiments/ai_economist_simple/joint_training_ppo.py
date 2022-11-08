import sys 

from gem.DQN_utils import save_models, load_models

from experiments.ai_economist_simple.elements import Agent
from experiments.ai_economist_simple.env import AIEcon_simple_game
from experiments.ai_economist_simple.env import generate_input, prepare_lstm, prepare_lstm2
# from examples.ai_economist_simple.Model_dualing_MLP import Model_linear_MLP_DDQN
import numpy as np
import torch
import random 
from experiments.ai_market_decider.simple_mlp import Model_simple_linear_MLP
from experiments.ai_economist_simple.PPO import RolloutBuffer, PPO 
# from examples.ai_economist_simple.market_SL import market
import itertools 



save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
#save_dir = "/Users/socialai/Dropbox/M1_ultra/"

device = "cpu"
print(device)


def create_models():
    models = []
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model1
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model2
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model3
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model1
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model2
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model3
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model1
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model2
    models.append(
        PPO(
            device=device, 
            state_dim=6,
            action_dim=7,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )
    )  # agent model3

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models

# AI_econ test game


# decider_model =  Model_simple_linear_MLP(
#             lr=0.0001,
#             replay_size=262144,  
#             in_size=18,  
#             hid_size1=10,  
#             hid_size2=10,  
#             out_size=2,
#             priority_replay=False,
#             device=device,
#         )

decider_model = PPO(
            device=device, 
            state_dim=18,
            action_dim=2,
            lr_actor=0.001,
            lr_critic=0.0005,
            gamma=0.9,
            K_epochs=10,
            eps_clip=0.2 
        )

# decider_model = load_models(save_dir, "decider_model")
decider_model.replay = RolloutBuffer() 

decider_model.model1.to(device)

models = create_models()
models.append(decider_model)

env = AIEcon_simple_game()

agent_list = []
i11, i12, i13, i14, i15 = 0, 0, 0, 0 ,0


binary = [0,1]

individual_attributes = [[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10] for i1 in binary for i2 in binary for i3 in binary for i4 in binary for i5 in binary for i6 in binary for i7 in binary for i8 in binary for i9 in binary for i10 in binary]
for indiv in individual_attributes:
    agent_type = np.random.choice([0,1,2])
    if agent_type == 0:
        agent_subtype = np.random.choice([0,1,2], p = (.50,.45,.05))
        appearence = [1, 0, 0] + indiv + [i11, i12, i13, i14, i15]
        if agent_subtype == 0:
            agent_list.append(Agent(0,0,appearence, .95, .15, .05))
        if agent_subtype == 1:
            agent_list.append(Agent(1,1,appearence, .15, .95, .05))
        if agent_subtype == 2:
            agent_list.append(Agent(2,2,appearence, .1, .1, .95))
    if agent_type == 1:
        agent_subtype = np.random.choice([0,1,2], p = (.45,.50,.05))
        appearence = [0, 1, 0] + indiv + [i11, i12, i13, i14, i15]
        if agent_subtype == 0:
            agent_list.append(Agent(3,3,appearence, .95, .15, .05))
        if agent_subtype == 1:
            agent_list.append(Agent(4,4,appearence, .15, .95, .05))
        if agent_subtype == 2:
            agent_list.append(Agent(5,5,appearence, .1, .1, .95))
    if agent_type == 2:
        agent_subtype = np.random.choice([0,1,2], p = (.1,.1,.8))
        appearence = [0, 0, 1] + indiv + [i11, i12, i13, i14, i15]
        if agent_subtype == 0:
            agent_list.append(Agent(6,6,appearence, .95, .15, .05))
        if agent_subtype == 1:
            agent_list.append(Agent(7,7,appearence, .15, .95, .05))
        if agent_subtype == 2:
            agent_list.append(Agent(8,8,appearence, .1, .1, .95))
    agent_list[-1].episode_memory = RolloutBuffer()

num_agents = len(agent_list)
print(num_agents)
# for i in range(num_agents):
#     agent_list[i].init_replay(3)   



rewards = [0,0,0,0,0,0,0,0,0]
losses = 0
decider_losses = 0
# model_learn_rate = 2
# sync_freq = 500

trainable_models = [0,1,2,3,4,5,6,7,8]
agent1_actions = [0,0,0,0,0,0,0]
agent2_actions = [0,0,0,0,0,0,0]
agent3_actions = [0,0,0,0,0,0,0]
agent4_actions = [0,0,0,0,0,0,0]
agent5_actions = [0,0,0,0,0,0,0]
agent6_actions = [0,0,0,0,0,0,0]
agent7_actions = [0,0,0,0,0,0,0]
agent8_actions = [0,0,0,0,0,0,0]
agent9_actions = [0,0,0,0,0,0,0]

decider_matrix = [0,0,0,0]

# epsilon = .99


# decider_step = 0

max_turns = 50

for epoch in range(1000000):
    done = 0

    env.wood = 10
    env.stone = 10

    for agent in range(len(agent_list)):
        agent_list[agent].coin = 0
        agent_list[agent].wood = 0
        agent_list[agent].stone = 0
        if agent_list[agent].policy == 2:
            agent_list[agent].coin = 6
        # agent_list[i].init_replay(3)
        # agent_list[i].state = torch.zeros(6).float()

    turn = 0
    while done != 1:
        turn = turn + 1
        if turn > max_turns:
            done = 1
        
        for agent in range(len(agent_list)):

            cur_wood = agent_list[agent].wood
            cur_stone = agent_list[agent].stone
            cur_coin = agent_list[agent].coin

            # state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)

            state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)
            # state_lstm = prepare_lstm(agent_list, agent, state)
            state = state.unsqueeze(0).to(device)
            # action= models[agent_list[agent].policy].take_action([state_lstm, epsilon])
            action, action_logprob = models[agent_list[agent].policy].take_action(state)
            env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [], agent_list, agent)


            if action in (3,4):
                decider_state = torch.tensor(agent_list[agent].appearance).float()
                decider_action, decider_action_logprob = decider_model.take_action(decider_state)
                agent_action = action - 3
                #reward = -1
                decider_reward = 1
                if decider_action != agent_action:
                    reward = -.1
                    decider_reward = -1
                if decider_action == agent_action and reward < .5:
                    decider_reward = -1
            
                if decider_action == 0 and agent_action == 0:
                    decider_matrix[0] = decider_matrix[0] + 1
                if decider_action == 1 and agent_action == 0:
                    decider_matrix[1] = decider_matrix[1] + 1
                if decider_action == 0 and agent_action == 1:
                    decider_matrix[2] = decider_matrix[2] + 1
                if decider_action == 1 and agent_action == 1:
                    decider_matrix[3] = decider_matrix[3] + 1

                # exp = (torch.tensor(agent_list[agent].appearance).float(), decider_action, decider_reward, torch.tensor(agent_list[agent].appearance).float(), done)
                # decider_model.replay.append(exp)

                # if decider_step % 20 == 0:
                #     decider_loss = decider_model.training(exp)
                #     decider_losses = decider_losses + decider_loss.detach().cpu().numpy()

                # if decider_step % 500 == 0 and decider_step > 4000:
                #     print(epoch, "decider maxtrx: ", decider_matrix, decider_losses, epsilon)
                #     epsilon = epsilon - .01

                decider_model.replay.states.append(decider_state)
                decider_model.replay.actions.append(decider_action)
                decider_model.replay.logprobs.append(decider_action_logprob)
                decider_model.replay.rewards.append(decider_reward)
                decider_model.replay.is_terminals.append(done)

                # if turn % max_turns == 0:
                #     print(epoch, "decider maxtrx: ", decider_matrix)

            agent_list[agent].episode_memory.states.append(state)
            agent_list[agent].episode_memory.actions.append(action)
            agent_list[agent].episode_memory.logprobs.append(action_logprob)
            agent_list[agent].episode_memory.rewards.append(reward)
            agent_list[agent].episode_memory.is_terminals.append(done)
            rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward

            # should make this a matrix to clean up code
            if agent_list[agent].policy == 0:
                agent1_actions[action] = agent1_actions[action] + 1
            if agent_list[agent].policy == 1:
                agent2_actions[action] = agent2_actions[action] + 1
            if agent_list[agent].policy == 2:
                agent3_actions[action] = agent3_actions[action] + 1

            if agent_list[agent].policy == 3:
                agent4_actions[action] = agent4_actions[action] + 1
            if agent_list[agent].policy == 4:
                agent5_actions[action] = agent5_actions[action] + 1
            if agent_list[agent].policy == 5:
                agent6_actions[action] = agent6_actions[action] + 1

            if agent_list[agent].policy == 6:
                agent7_actions[action] = agent7_actions[action] + 1
            if agent_list[agent].policy == 7:
                agent8_actions[action] = agent8_actions[action] + 1
            if agent_list[agent].policy == 8:
                agent9_actions[action] = agent9_actions[action] + 1

    
    for mods in trainable_models:
        shuffled_list = random.sample(agent_list, len(agent_list))
        for agent in shuffled_list: 
            if mods == agent.policy:
                loss = models[mods].training(agent.episode_memory, entropy_coefficient=0.01)
                agent.episode_memory.clear() 
                losses = losses + loss.detach().cpu().numpy()
    decider_loss = decider_model.training(decider_model.replay, entropy_coefficient=0.01)
    decider_model.replay.clear() 
    decider_losses = decider_losses + decider_loss.detach().cpu().numpy() 

    if epoch % round(500/max_turns) == 0:
        print("--------------------------------------")
        print("epoch:" , epoch, "loss: ",losses, "decider loss: ", decider_losses, "\n", "points (wood, stone, house): ", rewards)
        print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
        print("agent1 behaviours - chop_c: ", agent1_actions)
        print("agent2 behaviours - chop_m: ", agent2_actions)
        print("agent3 behaviours - chop_h: ", agent3_actions)
        print("agent4 behaviours - mine_c: ", agent4_actions)
        print("agent5 behaviours - mine_m: ", agent5_actions)
        print("agent6 behaviours - mine_h: ", agent6_actions)
        print("agent7 behaviours - hous_c: ", agent7_actions)
        print("agent8 behaviours - hous_m: ", agent8_actions)
        print("agent9 behaviours - hous_h: ", agent9_actions)
        print("decider maxtrx: ", decider_matrix)
        rewards = [0,0,0,0,0,0,0,0,0]
        losses = 0
        decider_losses = 0
        agent1_actions = [0,0,0,0,0,0,0]
        agent2_actions = [0,0,0,0,0,0,0]
        agent3_actions = [0,0,0,0,0,0,0]
        agent4_actions = [0,0,0,0,0,0,0]
        agent5_actions = [0,0,0,0,0,0,0]
        agent6_actions = [0,0,0,0,0,0,0]
        agent7_actions = [0,0,0,0,0,0,0]
        agent8_actions = [0,0,0,0,0,0,0]
        agent9_actions = [0,0,0,0,0,0,0]
        decider_matrix = [0,0,0,0]


    # if epoch % 10000 == 0:
    #     save_models(
    #     models,
    #     save_dir,
    #     "AIecon_simple_PPO_fullGame_MLP" + str(epoch),
    # )


