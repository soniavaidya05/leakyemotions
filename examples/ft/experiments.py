# --------------- #
# region: Imports #
# Import base packages
import os

from IPython.display import clear_output
from matplotlib import pyplot as plt
from typing import Optional, Union, Sequence

from examples.ft.env import FoodTrucks
from examples.ft.utils import visual_field, random_seed, set_seed
from examples.ft.config import (
    load_config,
    parse_args,
    create_agents,
    create_entities,
    create_models
)
# endregion       #
# --------------- #

def eval_model(file_name: str, 
               model_directory: Union[str, os.PathLike] = './checkpoints',
               n_games: int = 1, 
               show: bool = True,
               return_frames: bool = False,
               args: Union[Sequence[str], None] = None
               ) -> Union[list, None]:

    '''
    Load model parameters from disk and evaluate a N games with it.

    Parameters:
        file_name: The file name of the model. \n
        model_directory: The folder to search for the model in. \n
        n_games: The number of games to play. \n
        show: Whether to show the games using Pyplot. \n
        return_frames: Whether to return the game frames.\n
        args: (Optional) A sequence of strings in command line-like format 
        of game configuration parameters. Currently, only specifying the
        location of the config file via the flag `--config` is supported.

    Returns:
        (Optional) A list of lists of size n_games with the game frames.
    '''

    # Randomize seed
    seed = random_seed()
    
    # Read in configuration
    if args is not None:
        args = parse_args(command_line=False, args=args)
    else:
        args = parse_args(args)
    cfg = load_config(args)

    # Create game objects
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = FoodTrucks(cfg, agents, entities)

    # Load model checkpoint
    for agent in agents:
        agent.model.load(file_name = file_name,
                         dir = model_directory)
        # Zero out model epsilon
        agent.model.set_epsilon(0)

    total_points = 0

    # ----------------- #
    # region: Game loop #
    # ----------------- #
    for game in range(n_games):

        # Get the next seed
        set_seed(seed + game)

        # Game variables
        done, turn, points = 0, 0, 0

        ACTIONS = [
            'up', 'down', 'left', 'right'
        ]

        # Reset game objects
        env.reset()
        for agent in agents:
            agent.reset()

        if show:
            img = visual_field(env.world, return_rgb=True)
            clear_output(wait = True)
            plt.imshow(img)
            plt.title(f'Initial state.')
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

                # Show the outcome of the turn
                if show:
                    img = visual_field(env.world, return_rgb=True)
                    clear_output(wait = True)
                    plt.imshow(img)
                    plt.title(f'Game {seed}. Turn {turn}. Action: {ACTIONS[action]}. Reward: {reward}')
                    plt.show()
                    # print(f'Game {game}, turn {turn}. Agent moved {ACTIONS[action]} and received {reward} reward.')
        # ----------------- #
        # endregion         #
        # ----------------- #

        total_points += points

    # ----------------- #
    # endregion         #
    # ----------------- #
    
    print('')
    print(f'=' * 50)
    print(f'Final result of {n_games} games: Average return per game: {round(float(total_points) / n_games, 2)}.')
    print(f'=' * 50)


