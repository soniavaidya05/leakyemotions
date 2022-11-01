from examples.ai_economist_simple.elements_MLP import Agent
from examples.ai_economist_simple.env_MLP import AIEcon_simple_game
from examples.ai_economist_simple.env_MLP import generate_input, prepare_lstm, prepare_lstm2
from examples.ai_economist_simple.Model_dualing_MLP import Model_linear_MLP_DDQN
import numpy as np
import torch
from gem.DQN_utils import save_models, load_models, make_video
from examples.ai_market_decider.simple_mlp import Model_simple_linear_MLP



# note, the standard LSTM linear model was not working, so it was updated in this example folder
# that should be fixed


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
#save_dir = "/Users/socialai/Dropbox/M1_ultra/"

device = "cpu"
print(device)


def create_models():
    models = []
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model1
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model2
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model3
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model1
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model2
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model3
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model1
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
            out_size=7,
            priority_replay=False,
            device=device,
        )
    )  # agent model2
    models.append(
        Model_linear_MLP_DDQN(
            lr=0.0005,
            replay_size=4096,  
            in_size=18,  
            hid_size1=20,  
            hid_size2=20,  
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



decider_model = load_models(save_dir, "decider_model")
decider_model.replay = []

decider_model.model1.to(device)

models = create_models()
models.append(decider_model)

env = AIEcon_simple_game()

agent_list = []
i15 = 0
i14 = 0
i13 = 0
i12 = 0
i11 = 0

for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                for i4 in range(2):
                    for i5 in range(2):
                        for i6 in range(2):
                            for i7 in range(2):
                                for i8 in range(2):
                                    for i8 in range(2):
                                        for i9 in range(2):
                                            for i10 in range(2):
                                                #for i11 in range(2):
                                                    #for i12 in range(2):
                                                        #for i13 in range(2):
                                                            #for i14 in range(2):
                                                                #for i15 in range(2):
                                                                    agent_type = np.random.choice([0,1,2])
                                                                    if agent_type == 0:
                                                                        agent_subtype = np.random.choice([0,1,2], p = (.45,.45,.1))
                                                                        appearence = [1, 0, 0, i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]
                                                                        if agent_subtype == 0:
                                                                            agent_list.append(Agent(0,0,appearence, .95, .15, .05))
                                                                        if agent_subtype == 1:
                                                                            agent_list.append(Agent(1,1,appearence, .95, .15, .05))
                                                                        if agent_subtype == 2:
                                                                            agent_list.append(Agent(2,2,appearence, .95, .15, .05))
                                                                    if agent_type == 1:
                                                                        agent_subtype = np.random.choice([0,1,2], p = (.45,.45,.1))
                                                                        appearence = [0, 1, 0, i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]
                                                                        if agent_subtype == 0:
                                                                            agent_list.append(Agent(3,3,appearence, .95, .15, .05))
                                                                        if agent_subtype == 1:
                                                                            agent_list.append(Agent(4,4,appearence, .95, .15, .05))
                                                                        if agent_subtype == 2:
                                                                            agent_list.append(Agent(5,5,appearence, .95, .15, .05))
                                                                    if agent_type == 2:
                                                                        agent_subtype = np.random.choice([0,1,2], p = (.1,.1,.8))
                                                                        appearence = [0, 0, 1, i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]
                                                                        if agent_subtype == 0:
                                                                            agent_list.append(Agent(6,6,appearence, .95, .15, .05))
                                                                        if agent_subtype == 1:
                                                                            agent_list.append(Agent(7,7,appearence, .95, .15, .05))
                                                                        if agent_subtype == 2:
                                                                            agent_list.append(Agent(8,8,appearence, .95, .15, .05))


num_agents = len(agent_list)
print(num_agents)
for i in range(num_agents):
    agent_list[i].init_replay(3)   



rewards = [0,0,0,0,0,0,0,0,0]
losses = 0
model_learn_rate = 2
sync_freq = 500

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
        agent_list[agent].coin = 0
        agent_list[agent].wood = 0
        agent_list[agent].stone = 0
        if agent_list[agent].policy == 2:
            agent_list[agent].coin = 6
        agent_list[i].init_replay(3)
        agent_list[i].state = torch.zeros(6).float()

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

            state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)
            state_lstm = prepare_lstm(agent_list, agent, state)
            state = state.unsqueeze(0).to(device)
            action= models[agent_list[agent].policy].take_action([state_lstm, epsilon])
            env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [], agent_list, agent)
            rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward



            if action in (3,4):
                decider_state = torch.tensor(agent_list[agent].appearance).float()
                decider_action = decider_model.take_action([decider_state, .1])
                agent_action = action - 3
                decider_reward = -1
                if decider_action != agent_action:
                    reward = -.1


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

        if turn % model_learn_rate == 0 and epoch > 100:
            for mods in trainable_models:
                loss = models[mods].training(128, .9) 
                losses = losses + loss.detach().cpu().numpy()

    if turn % model_learn_rate == 0:
        for mods in trainable_models:# reducing gamma to see if future Q is the problem
            loss = models[mods].training(128, .9) 
            losses = losses + loss.detach().cpu().numpy()

    if epoch % round(500/max_turns) == 0:
        print("--------------------------------------")
        print("epoch:" , epoch, "loss: ",losses, "points (wood, stone, house): ", rewards, "epsilon: ", epsilon)
        print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
        print("agent1 behaviours - chop_c: ", agent1_actions)
        print("agent2 behaviours - chop_m: ", agent2_actions)
        print("agent3 behaviours - chop_h: ", agent3_actions)
        print("agent4 behaviours - mine_c: ", agent1_actions)
        print("agent5 behaviours - mine_m: ", agent2_actions)
        print("agent6 behaviours - mine_h: ", agent3_actions)
        print("agent7 behaviours - hous_c: ", agent1_actions)
        print("agent8 behaviours - hous_m: ", agent2_actions)
        print("agent9 behaviours - hous_h: ", agent3_actions)
        rewards = [0,0,0,0,0,0,0,0,0]
        losses = 0
        agent1_actions = [0,0,0,0,0,0,0]
        agent2_actions = [0,0,0,0,0,0,0]
        agent3_actions = [0,0,0,0,0,0,0]
        agent4_actions = [0,0,0,0,0,0,0]
        agent5_actions = [0,0,0,0,0,0,0]
        agent6_actions = [0,0,0,0,0,0,0]
        agent7_actions = [0,0,0,0,0,0,0]
        agent8_actions = [0,0,0,0,0,0,0]
        agent9_actions = [0,0,0,0,0,0,0]

    if epoch % 10000 == 0:
        save_models(
        models,
        save_dir,
        "AIecon_simple_dualing_fullGame_MLP" + str(epoch),
    )



