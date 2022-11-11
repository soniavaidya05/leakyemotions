from experiments.ai_economist_simple.elements import Agent
from experiments.ai_economist_simple.env import AIEcon_simple_game
from experiments.ai_economist_simple.env import generate_input, prepare_lstm, prepare_lstm2
from experiments.ai_economist_simple.old.Model_dualing_MLP import Model_linear_MLP_DDQN
import numpy as np
import torch
from gem.DQN_utils import save_models, load_models, make_video



# note, the standard LSTM linear model was not working, so it was updated in this example folder
# that should be fixed


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"


device = "cpu"
print(device)


def create_models():
    models = []
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=1024,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )
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
    # do all the agents need to have a unique object?
    pass

rewards = [0,0,0]
losses = 0
model_learn_rate = 2
sync_freq = 500

trainable_models = [0]

epsilon = .99



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
        epsilon = epsilon - .0001

    env.wood = 10
    env.stone = 10

    for agent in range(len(agent_list)):
        # reset the agents for a new epoch
        agent_list[i].init_replay(3)
        agent_list[i].state = torch.zeros(12).float()

    turn = 0
    while done != 1:
        turn = turn + 1
        if turn > max_turns:
            done = 1
        for agent in range(len(agent_list)):
            """
            This is where some thought will need to go in, if the agents are in lists in locations
            """

            state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)
            state_lstm = prepare_lstm(agent_list, agent, state)
            state = state.unsqueeze(0).to(device)
            action= models[agent_list[agent].policy].take_action([state_lstm, epsilon])
            env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [], agent_list, agent)
            rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward
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
                loss = models[mods].training(128, .9) 
                losses = losses + loss.detach().cpu().numpy()

    if turn % model_learn_rate == 0:
        for mods in trainable_models:# reducing gamma to see if future Q is the problem
            losses = losses + loss.detach().cpu().numpy()
