# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import argparse

from IPython.display import clear_output
from matplotlib import pyplot as plt
from typing import Union, Sequence

# Import gem-specific packages
from agentarium.utils import ( 
    random_seed, 
    set_seed, 
    fig2img, 
    nearest_2_power, 
    minmax,
)
from agentarium.visual_field import visual_field
from agentarium.buffers import ActionBatchReplayBuffer as Buffer
from examples.trucks.env import FoodTrucks
from examples.trucks.utils import (
    color_map,
    GameVars
)
from examples.trucks.config import (
    create_agents,
    create_entities,
    create_models
)
from agentarium.config import load_config

# --------------- #
# endregion       #
# --------------- #

def eval_model(*flags,
               model_path: Union[str, os.PathLike],
               config_path: Union[str, os.PathLike],
               n_games: int = 1,
               ) -> list:

    '''
    Load model parameters from disk and evaluate N games with it.

    Parameters:
        *flags: An unpacked list of game objects to return. 
        Currently implemented: `memories`, `frames`, `scores`. 
        `jupyter-mode` modifies the pretty print function for ipynb output.
        model_path: The path to the model weights. \n
        config_path: The path to the configuration file. \n
        n_games: The number of games to play. \n

    Returns:
        A dictionary of game objects based on the flags passed in. 
    '''

    # Initialize the dictionary of game objects based on the flags passed in.
    game_objects = dict.fromkeys(flags)

    # Randomize seed
    seed = random_seed()
    
    # Read in configuration
    args = argparse.Namespace(config=config_path)
    cfg = load_config(args)

    # Create game objects
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = FoodTrucks(cfg, agents, entities)

    # Load model checkpoint
    for agent in agents:
        agent.model.load(model_path)
        # Zero out model epsilon
        agent.model.set_epsilon(0)
        # Evaluation mode
        agent.model.eval()

    # Create experience buffer for the inverse model
    if 'memories' in flags:
        # Use the nearest power of 2 for the buffer size and batch size
        buffer_size = nearest_2_power(cfg.experiment.max_turns * n_games)
        # Batch size cannot be smaller than 64 or greater than 1024
        batch_size = minmax(buffer_size // 16, 64, 1024)
        buffer = Buffer(
            buffer_size=buffer_size,
            batch_size=batch_size, 
            device=cfg.model.iqn.device,
            seed=seed,
            gamma=cfg.model.iqn.parameters.GAMMA,
            timesteps=cfg.model.iqn.parameters.num_frames,
            action_space=5
        )
    # Create a list to add frames to
    if 'frames' in flags:
        game_objects['frames'] = []
    scores = GameVars(max_epochs=n_games)

    # ----------------- #
    # region: Game loop #
    # ----------------- #
    for game in range(n_games):

        # Get the next seed
        set_seed(seed + game)

        # Game variables
        done, turn, points = 0, 0, 0
        frames = []
        ACTIONS = [
            'up', 'down', 'left', 'right'
        ]

        # Reset game objects
        env.reset()
        for agent in agents:
            agent.reset()

        if 'frames' in flags:
            img = visual_field(env.world, color_map, return_rgb=True)
            clear_output(wait = True)
            print(f'╔═════════════════════════════════════════════════════╗')
            print(f'║                 Initial game state...               ║')
            print(f'╚═════════════════════════════════════════════════════╝')
            plt.imshow(img)
            plt.title(f'Initial state.')
            frame = fig2img(plt)
            frames.append(frame)
            plt.show()
        
        # ----------------- #
        # region: Turn loop #
        # ----------------- #
        while done == 0:
            
            # Increment turn counter
            turn += 1

            for agent in agents:

                # Agent transition
                state, action, reward, next_state, done_ = agent.transition(env)
                points += reward

                # If agent finished or max turns is reached, end the turn
                if turn >= cfg.experiment.max_turns or done_: 
                    done = 1 # End loop if maximum turns have been reached
                
                # Append the experience to the replay
                exp = (0, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                if 'memories' in flags:
                    buffer.add(state, action, reward, next_state, done)

                # Show the outcome of the turn
                if 'frames' in flags:
                    img = visual_field(env.world, return_rgb=True)
                    clear_output(wait = True)
                    scores.pretty_print(*flags, epoch=game, turn=turn, reward=points)
                    plt.imshow(img)
                    plt.title(f'Game {seed}. Turn {turn}. Action: {ACTIONS[action]}. Reward: {reward}')
                    frame = fig2img(plt)
                    frames.append(frame)
                    plt.show()
        
        # ----------------- #
        # endregion         #
        # ----------------- #

        # Collect information for the scores and the frames, and reset the batched action deque.
        if 'memories' in flags:
            buffer.reset_action_batch()
        if 'scores' in flags:
            scores.record_turn(game, turn, 0., points)
        if 'frames' in flags:
            game_objects['frames'].append(frames)    

    # ----------------- #
    # endregion         #
    # ----------------- #

    # After the game is over, add the scores and the buffer to the game objects
    if 'memories' in flags:
        game_objects['memories'] = buffer
    if 'scores' in flags:
        game_objects['scores'] = scores

    # Remove `jupyter-mode` from the dictionary, as we don't need to return it.
    game_objects.pop('jupyter-mode', None)

    return game_objects

def train_inverse_model(model, plot_freq = 100, **kwargs):
        '''
        Defines the training loop for the inverse model.
        '''
        
        losses = []

        for i in range(10000):

            loss = model.train_model()
            losses.append(loss)
            if i % 100 == 0:
                
                # If an experiment configuration has been passed in...
                if 'cfg' in kwargs:
                    models = create_models(kwargs['cfg'])
                    agents = create_models(kwargs['cfg'], models)
                    entities = create_entities(kwargs['cfg'])

                    env = FoodTrucks(kwargs['cfg'], agents, models)