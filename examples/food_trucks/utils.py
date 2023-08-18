from collections import deque
from IPython.display import clear_output
from gem.utils import find_instance
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from gem.models.perception import agent_visualfield
import random
from examples.RPG3_PPO.PPO_CPC import (
    RolloutBuffer
)
from PIL import Image
import numpy as np
import torch
from IPython.display import clear_output

def generate_memories(models, env, n_games = 10000, show = False):
    max_turns = 100
    world_size = 11
    memories = deque(maxlen=max_turns*n_games)
    total_reward = 0
    memories_added = 0
    for i in range(n_games):
        clear_output(wait = True)
        print(f'Running game {i}...')

        env.reset_env()

        done = 0
        turn = 0

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            agent = env.world[loc]
            agent.init_replay(
                numberMemories=2,
            )
            agent.init_rnn_state = None

        while done == 0:
            turn += 1
            agentList = find_instance(env.world, "neural_network")
            for loc in agentList:

                agent = env.world[loc]
                state = env.pov(
                    loc, 
                )
               

                action = models[agent.policy].take_action(state, 0)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc
                ) = agent.transition(env, models, action, loc)

                total_reward += reward

                if turn >= max_turns:
                    done = 1

                if reward > 0:
                    memories_added +=1 
                    memories.append(
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done
                        )
                    )

    print(f'Memories added: {memories_added}. Average reward obtained per game: {round(float(total_reward)/float(memories_added), 2)}')
                
    return memories

def eval_game(models, env, turn, epsilon, epochs=10000, max_turns=100, filename="tmp"):
    """
    This is the main loop of the game
    """
    game_points = [0, 0]

    fig = plt.figure()
    ims = []
    """
    Move each agent once and then update the world
    Creates new gamepoints, resets agents, and runs one episode
    """

    done = 0

    # create a new gameboard for each epoch and repopulate
    # the resset does allow for different params, but when the world size changes, odd
    env.reset_env(
        height=env.height,
        width=env.width,
        layers=1,
    )

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        agent = env.world[loc] # Agent at this location
        agent.init_replay(1)
        agent.init_rnn_state = None

    for _ in range(max_turns):
        """
        Find the agents and wolves and move them
        """

        image = agent_visualfield(env.world, (0, 0), env.tile_size, k=None)
        im = plt.imshow(image, animated=True)
        ims.append([im])

        agentList = find_instance(env.world, "neural_network")

        random.shuffle(agentList)

        for loc in agentList:
            agent = env.world[loc] # Agent at this location
            agent.reward = 0 # Reset the agent's reward
            device = models[agent.policy].device
            state = env.pov(loc)
            params = (state.to(device), epsilon, agent.init_rnn_state)

            # set up the right params below

            action = models[agent.policy].take_action(state, 0)

            (
                env.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = agent.transition(env, models, action[0], loc)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)

def generate_memories(models, env, n_games = 10000, show = False):
    max_turns = 100
    world_size = 11
    memories = deque(maxlen=max_turns*n_games)
    for i in range(n_games):
        clear_output(wait = True)
        print(f'Running game {i}...')

        env.reset_env()

        done = 0
        turn = 0

        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            agent = env.world[loc]
            agent.init_replay(
                numberMemories=2,
            )
            agent.init_rnn_state = None

        while done == 0:
            turn += 1
            agentList = find_instance(env.world, "neural_network")
            for loc in agentList:

                agent = env.world[loc]
                state = env.pov(
                    loc, 
                )
               

                action = models[agent.policy].take_action(state, 0)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc
                ) = agent.transition(env, models, action, loc)

                if turn >= max_turns:
                    done = 1
                memories.append(
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done
                    )
                )
    return memories

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def viz(env, text = False):
    if text:
        for i in range(env.height):
            print(f'{" ".join([str(j)[0] for j in env.world[i, :, 0]])}')
    else:
        map = np.zeros((env.height, env.width, 3), dtype=np.uint8)
        temp = env.one_hot # temporarily store one hot parameter
        env.one_hot = False # set to false for pixel view
        colors = env.color_map() # change colours to pixel view
        for i in range(env.world.shape[0]):
            for j in range(env.world.shape[1]):
                obj_type = str(env.world[i, j, 0]).split('_')
                map[i, j, :] = colors.get(obj_type[0] + '_color')
        env.one_hot = temp
        plt.imshow(map)
        plt.show()

def run_one_game(
        models,
        env,
        max_turns=100,
        text = False
    ):

    done, turn = 0, 0
    env.reset_env()

    viz(env)

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        agent = env.world[loc] # Agent at this location
        agent.init_replay(
            2,
            one_hot = env.one_hot
        )
        agent.init_rnn_state = None

    while done == 0:
        
        clear_output(wait = True)
        turn+=1
        agentList = find_instance(env.world, "neural_network")
        print(f'Agent location: {agentList[0]}')
        random.shuffle(agentList)

        # For each agent, act and get the transitions and experience
        for loc in agentList:
            agent = env.world[loc]
            agent.reward = 0
            state = env.pov(loc)

            print(state[0, 1, 1, :, :])

            with torch.no_grad():
                action = models[agent.policy].take_action(state, 0.05)

            (
                env.world,
                reward,
                next_state,
                done,
                new_loc,
            ) = agent.transition(env, models, action[0], loc)

            movements = ['up', 'down', 'left','right']

            
            print(f'Taking action {movements[action[0]]}.')
            viz(env, text)

        # determine whether the game is finished (either max length or all agents are dead)
        if turn > max_turns:
            done = 1


def run_games(
        models,
        env,
        max_turns = 100,
        epochs = 10
    ):

    game_points = [0, 0]
    gems = [0, 0, 0, 0]

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        viz(env)

        done, withinturn = 0, 0

        # create a new gameboard for each epoch and repopulate
        # the reset does allow for different params, but when the world size changes, odd
        env.reset_env()
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            agent = env.world[loc] # Agent at this location
            agent.init_replay(
                2,
                one_hot = env.one_hot
            )
            agent.init_rnn_state = None

        while done == 0:
            """
            While the agent is not done, move the agent
            """
            withinturn = withinturn + 1

            # Find agents in the environment; they move in a randomized order
            agentList = find_instance(env.world, "neural_network")
            print(f'Agent loc: {agentList[0]}')
            random.shuffle(agentList)

            # For each agent, act and get the transitions and experience
            for loc in agentList:
                agent = env.world[loc]
                agent.reward = 0
                state = env.pov(loc)

                action = models[agent.policy].take_action(state, 0)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                ) = agent.transition(env, models, action[0], loc)

                movements = ['up', 'down', 'left','right']

                
                clear_output(wait = True)
                print(f'Taking action {movements[action[0]]}.')
                viz(env)

                if reward == 10:
                    gems[0] = gems[0] + 1
                if reward == 5:
                    gems[1] = gems[1] + 1
                if reward == -5:
                    gems[2] = gems[2] + 1
                if reward == -1:
                    gems[3] = gems[3] + 1

                # these can be included on one replay

                exp = (
                    # models[env.world[new_loc].policy].max_priority,
                    1,
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    ),
                )

                env.world[new_loc].episode_memory.append(exp)

                if env.world[new_loc].kind == "agent":
                    game_points[0] = game_points[0] + reward

            # determine whether the game is finished (either max length or all agents are dead)
            if (
                withinturn > max_turns
                or len(find_instance(env.world, "neural_network")) == 0
                or reward in env.truck_prefs
            ):
                done = 1
