from examples.ai_economist_simple.elements import Agent
from examples.ai_economist_simple.env import AIEcon_simple_game
from examples.ai_economist_simple.env import generate_input, prepare_lstm, prepare_lstm2
from examples.ai_economist_simple.Model_dualing_NoneLSTM import Model_linear_LSTM_DQN
import numpy as np
import torch
from gem.DQN_utils import save_models, load_models, make_video

from examples.ai_market_decider.simple_mlp import Model_simple_linear_MLP


# note, the standard LSTM linear model was not working, so it was updated in this example folder
# that should be fixed


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"


device = "cpu"
print(device)


decider_model =  Model_simple_linear_MLP(
            lr=0.0001,
            replay_size=262144,  
            in_size=18,  
            hid_size1=10,  
            hid_size2=10,  
            out_size=2,
            priority_replay=False,
            device=device,
        )




decider_model.model1.to(device)



# AI_econ test game


#models = create_models()

models = load_models(save_dir, "AIecon_simple_dualing_180000")

env = AIEcon_simple_game()

agent_list = []
num_agents = 32768
for i in range(num_agents):
    agent_type = np.random.choice([0,1,2])
    if agent_type == 0:
        groupview = np.array([1,0,0])
        individ = np.random.choice([0, 1], size=(15,))
        appearence = np.concatenate([groupview, individ])
        agent_list.append(Agent(0,0,appearence, .95, .15, .05))
    if agent_type == 1:
        groupview = np.array([0,1,0])
        individ = np.random.choice([0, 1], size=(15,))
        appearence = np.concatenate([groupview, individ])
        agent_list.append(Agent(1,1,appearence, .15, .95, .05))
    if agent_type == 2:
        groupview = np.array([0,0,1])
        individ = np.random.choice([0, 1], size=(15,))
        appearence = np.concatenate([groupview, individ])        
        agent_list.append(Agent(2,2,appearence, .05, .05, .15))
    agent_list[i].init_replay(3)   

rewards = [0,0,0]
losses = 0
decider_losses = 0
model_learn_rate = 2
sync_freq = 500

trainable_models = [0,1,2]
agent1_actions = [0,0,0,0,0,0,0]
agent2_actions = [0,0,0,0,0,0,0]
agent3_actions = [0,0,0,0,0,0,0]

decider_matrix = [0,0,0,0]


epsilon = .98

decider_step = 0



max_turns = 50

for epoch in range(1000000):
    done = 0
    if epoch % round(sync_freq/max_turns) == 0:
            # update the double DQN model ever sync_frew
            for mods in trainable_models:
                models[mods].model2.load_state_dict(
                    models[mods].model1.state_dict()
                )
    #if epoch % round(1000/max_turns) == 0:
    #    epsilon = epsilon - .1

    env.wood = 10
    env.stone = 10

    for agent in range(len(agent_list)):
        agent_list[agent].coin = 0
        agent_list[agent].wood = 0
        agent_list[agent].stone = 0
        if agent_list[agent].policy == 2:
            agent_list[agent].coin = 6
        agent_list[i].init_replay(3)
        agent_list[i].state = torch.zeros(12).float()

    turn = 0
    while done != 1:
        turn = turn + 1
        if turn > max_turns:
            done = 1
        
        for agent in range(len(agent_list)):

            cur_wood = agent_list[agent].wood
            cur_stone = agent_list[agent].stone
            cur_coin = agent_list[agent].coin

            state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)

            state = state.unsqueeze(0).to(device)
            state_lstm = prepare_lstm(agent_list, agent, state)
            action, init_rnn_state= models[agent_list[agent].policy].take_action([state_lstm, .1, agent_list[agent].init_rnn_state])
            agent_list[agent].init_rnn_state = init_rnn_state
            #print(action)
            env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [], agent_list, agent)
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
                agent_list[agent].init_rnn_state
            )]

            if action in (3,4):
                decider_step = decider_step + 1
                decider_state = torch.tensor(agent_list[agent].appearance).float()
                decider_action = decider_model.take_action([decider_state, epsilon])
                agent_action = action - 3
                decider_reward = -1
                if decider_action == agent_action:
                    decider_reward = 1

                if decider_action == 0 and agent_action == 0:
                    decider_matrix[0] = decider_matrix[0] + 1
                if decider_action == 1 and agent_action == 0:
                    decider_matrix[1] = decider_matrix[1] + 1
                if decider_action == 0 and agent_action == 1:
                    decider_matrix[2] = decider_matrix[2] + 1
                if decider_action == 1 and agent_action == 1:
                    decider_matrix[3] = decider_matrix[3] + 1


                exp = (torch.tensor(agent_list[agent].appearance).float(), decider_action, decider_reward, torch.tensor(agent_list[agent].appearance).float(), done)
                decider_model.replay.append(exp)

                if decider_step % 2 == 0:
                    decider_loss = decider_model.training(exp)
                    decider_losses = decider_losses + decider_loss.detach().cpu().numpy()

                if decider_step % 5000 == 0 and decider_step > 40000:
                    print(epoch, "decider maxtrx: ", decider_matrix, decider_losses, epsilon)
                    epsilon = epsilon - .01

            #agent_list[agent].episode_memory.append(exp)

            #models[agent_list[agent].policy].PER_replay.add(exp[0], exp[1])

        #if turn % model_learn_rate == 0:
        #    for mods in trainable_models:
        #        loss = models[mods].training(128, .9) 
        #        losses = losses + loss.detach().cpu().numpy()

    #if turn % model_learn_rate == 0:
    #    for mods in trainable_models:# reducing gamma to see if future Q is the problem
    #        losses = losses + loss.detach().cpu().numpy()

    if epoch % round(500/max_turns) == 0:
        print("--------------------------------------")
        print("epoch:" , epoch, "loss: ",losses, "points (wood, stone, house): ", rewards, "epsilon: ", epsilon)
        print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
        print("agent1 behaviours: ", agent1_actions)
        print("agent2 behaviours: ", agent2_actions)
        print("agent3 behaviours: ", agent3_actions)
        print("decider maxtrx: ", decider_matrix)
        rewards = [0,0,0]
        losses = 0
        decider_losses = 0
        agent1_actions = [0,0,0,0,0,0,0]
        agent2_actions = [0,0,0,0,0,0,0]
        agent3_actions = [0,0,0,0,0,0,0]
        decider_matrix = [0,0,0,0]


    #if epoch % 10000 == 0:
    #    save_models(
    #    models,
    #    save_dir,
    #    "AIecon_simple_dualing_None_" + str(epoch),
    #)



