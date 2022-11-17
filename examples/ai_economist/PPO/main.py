from gem.utils import (
    update_epsilon,
    update_memories,
    find_moveables,
    transfer_world_memories,
    find_instance,
    update_memories_rnn,
)
from examples.ai_economist.elements import (
    Agent,
    Wood,
    Stone,
    House,
    EmptyObject,
    Wall,
)
from examples.ai_economist.PPO.cnn_PPO import PPO, RolloutBuffer
from examples.ai_economist.env import AI_Econ
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import torch.nn as nn
import torch.nn.functional as F
from gem.DQN_utils import save_models, load_models, make_video2

import torch

import random

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"
# save_dir = "/Users/socialai/Dropbox/M1_ultra/"
# save_dir = "C:/Users/wilcu/OneDrive/Documents/gemout/"

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if torch.backends.mps.is_available():
#    device = torch.device("mps")

print(device)


def create_models():
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """
    models = []
    models.append(
        PPO(
            device=device,
            state_dim=1300,
            action_dim=5,
            lr_actor=0.0001,  # .001
            lr_critic=0.0005,  # .0005
            gamma=0.92,  # was .9
            K_epochs=10,  # was 10
            eps_clip=0.2,
        )
    )  # agent model 1

    models.append(
        PPO(
            device=device,
            state_dim=1300,
            action_dim=5,
            lr_actor=0.0001,  # .001
            lr_critic=0.0005,  # .0005
            gamma=0.92,  # was .9
            K_epochs=10,  # was 10
            eps_clip=0.2,
        )
    )  # agent model 2

    models.append(
        PPO(
            device=device,
            state_dim=1300,
            action_dim=5,
            lr_actor=0.0001,  # .001
            lr_critic=0.0005,  # .0005
            gamma=0.92,  # was .9
            K_epochs=10,  # was 10
            eps_clip=0.2,
        )
    )  # agent model 3

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models


def fix_next_state(state, next_state):
    state_ = state.clone()
    state_[:, 0:-1, :, :, :] = state[:, 1:, :, :, :]
    state_[:, -1, :, :, :] = next_state[:, -1, :, :, :]
    return state_


world_size = 30

trainable_models = [0, 1, 2]
sync_freq = 500
modelUpdate_freq = 5
epsilon = 0.99

turn = 1

models = create_models()
env = AI_Econ(
    height=world_size,
    width=world_size,
    layers=2,
    defaultObject=EmptyObject(),
    wood1p=0.04,
    stone1p=0.04,
)
env.game_test()


def run_game(
    models,
    env,
    turn,
    epsilon,
    epochs=10000,
    max_turns=100,
):
    """
    This is the main loop of the game
    """
    losses = 0
    game_points = [0, 0, 0]
    for epoch in range(epochs):

        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the resset does allow for different params, but when the world size changes, odd
        env.reset_env(
            height=world_size,
            width=world_size,
            layers=2,
            wood1p=0.04,
            stone1p=0.04,
        )
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(3)
            env.world[loc].reward = 0
            env.world[loc].wood = 0
            env.world[loc].stone = 0
            env.world[loc].labour = 100
            env.world[loc].coin = 4
            env.world[loc].episode_memory_PPO = RolloutBuffer()


        while done == 0:
            """
            Find the agents and wolves and move them
            """
            turn = turn + 1
            withinturn = withinturn + 1

            #if epoch % sync_freq == 0:
            #    # update the double DQN model ever sync_frew
            #    for mods in trainable_models:
            #        models[mods].model2.load_state_dict(
            #            models[mods].model1.state_dict()
            #        )

            use_labour = False
            if use_labour == True:
                # if using labour, then the agents will only be able to move if they have labour
                # this is a way to make the agents more conservative
                agentList = find_instance(env.world, "neural_network")
                for loc in agentList:
                    if env.world[loc].labour < 0:
                        env.world[loc].static = 1
                        env.world[loc].trainable = 0
                        env.world[loc].has_transitions = False
                        env.world[loc].episode_memory_PPO = RolloutBuffer()

            # note, we need to set it up so that once an agent runs out of labour, it can't move

            agentList = find_instance(env.world, "neural_network")
            for loc in agentList:
                # reset reward for the turn
                env.world[loc].reward = 0
            random.shuffle(agentList)

            for loc in agentList:
                """
                Reset the rewards for the trial to be zero for all agents
                """
                env.world[loc].reward = 0

            for loc in agentList:
                if env.world[loc].static != 1:

                    holdObject = env.world[loc]

                    device = models[holdObject.policy].device
                    state = env.pov(
                        world=env.world,
                        location=loc,
                        holdObject=env.world[loc],
                        inventory=[
                            float(env.world[loc].stone),
                            float(env.world[loc].wood),
                            float(env.world[loc].coin),
                        ],
                        layers=[0, 1],
                    )

                    action, action_logprob, init_rnn_state = models[
                        env.world[loc].policy
                    ].take_action(state, env.world[loc].init_rnn_state)
                    env.world[loc].init_rnn_state = init_rnn_state


                    (
                        env.world,
                        reward,
                        next_state,
                        done,
                        new_loc,
                    ) = holdObject.transition(env, models, action, loc)


                    next_state = fix_next_state(state, next_state)

                    exp = 1, (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        #env.world[new_loc].init_rnn_state[0],
                        #env.world[new_loc].init_rnn_state[1],
                    )
                    env.world[new_loc].episode_memory_PPO.states.append(state)
                    env.world[new_loc].episode_memory_PPO.actions.append(action)
                    env.world[new_loc].episode_memory_PPO.logprobs.append(
                        action_logprob
                    )

                    env.world[new_loc].episode_memory_PPO.rewards.append(reward)
                    env.world[new_loc].episode_memory_PPO.is_terminals.append(done)
                    env.world[new_loc].episode_memory.append(exp)




                    if env.world[new_loc].kind == "agent":
                        game_points[env.world[new_loc].policy] = game_points[env.world[new_loc].policy] + reward

            # note that with the current setup, the world is not generating new wood and stone
            # we will need to consider where to add the transitions that do not have movement or neural networks
            regenList = find_instance(env.world, "deterministic")

            for loc in regenList:
                env.world = env.world[loc].transition(env.world, loc)

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
            ):
                done = 1

            if len(trainable_models) > 0:
                """
                Update the next state and rewards for the agents after all have moved
                And then transfer the local memory to the model memory
                """
                # this updates the last memory to be the final state of the game board
                env.world = update_memories(
                    env, find_moveables(env.world), done, end_update=False
                )

                # transfer the events for each agent into the appropriate model after all have moved
                #models = transfer_world_memories(
                #    models, env.world, find_moveables(env.world), extra_reward=False
                #)

            #if withinturn % modelUpdate_freq == 0:
            #    """
            #    Train the neural networks within a eposide at rate of modelUpdate_freq
            #    """
            #    for mods in trainable_models:
            #        loss = models[mods].training(64, 0.9)
            #        losses = losses + loss.detach().cpu().numpy()

        #for mods in range(len(trainable_models)):

        agentList = find_instance(env.world, "neural_network")
        """
        Train the neural networks at the end of eac epoch
        """
        for loc in agentList:
            #print(env.world[loc].episode_memory_PPO.rewards)
            loss = models[env.world[loc].policy].training(
                env.world[loc].episode_memory_PPO, entropy_coefficient=0.005
            )  # entropy_coefficient=0.01 was before
            env.world[loc].episode_memory_PPO.clear()
            losses = losses + loss.detach().cpu().numpy()


        updateEps = False
        # TODO: the update_epsilon often does strange things. Needs to be reconceptualized
        if updateEps == True:
            epsilon = update_epsilon(epsilon, turn, epoch)

        if epoch % 10 == 0 and len(trainable_models) > 0:
            # print the state and update the counters. This should be made to be tensorboard instead
            print(
                epoch,
                withinturn,
                round(game_points[0]),round(game_points[1]),round(game_points[2]),
                losses,
                epsilon,
            )
            game_points = [0, 0, 0]
            losses = 0
    return models, env, turn, epsilon


models = create_models()


run_params = ([0.9, 100000, 100, 31],)


# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
    )
    save_models(models, save_dir, "AI_econ_PER" + str(modRun))
    make_video2("AI_econ_PER" + str(modRun), save_dir, models, 30, env)


# models = load_models(save_dir, "AI_econ_test28")
# make_video2("test_new_priority", save_dir, models, 30, env)



