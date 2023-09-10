# ------------------------ #
# region: Imports          #
import sys
import os

# ------------------------ #
# region: path nonsense    #
# Determine appropriate paths for imports and storage
current_path = os.path.abspath('.').split('/')
# If we are already in the base directory...
if current_path[-1] == 'transformers':
    root = os.path.abspath('.')
# From the ft directory, we need to go up two folders
elif current_path[-1] == 'ft':
    root = os.path.abspath('../..')
# Base case: specify the folder manually
else:
    root = os.path.abspath('/Users/rgelpi/Documents/GitHub/transformers') # Change the wd as needed.

# Make sure the transformers directory is in PYTHONPATH
if root not in sys.path:
    sys.path.insert(0, root)
# endregion                #
# ------------------------ #

# Import base packages
import torch
import numpy as np
import random
import argparse
from datetime import datetime

# Import gem-specific packages
from examples.ft.env import FoodTrucks
from examples.ft.config import create_models, create_agents, create_entities, load_config, init_log, Cfg
from examples.ft.utils import GameVars

# endregion                #
# ------------------------ #

# ------------------------------------------ #
# region: Set seeds and global print options #
SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
# endregion
# ------------------------------------------ #

def run_game(
        cfg: Cfg,
        **kwargs
    ) -> None:
    '''
    Run a game according to the experiment configuration.

    Parameters:
        cfg: The configuration object
    '''
    # Create model objects and environment
    init_log(cfg)
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = FoodTrucks(cfg, agents, entities)
    
    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            log_dir=f'{root}/examples/ft/runs/{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameVars(cfg.experiment.epochs)

    # If a path to a model is specified in the run, load those weights
    if 'load_weights' in kwargs:
        for agent in agents:
            agent.model.load(file_path = kwargs.get('load_weights'))

    # Set the custom epsilon if it is specified in the kwargs
    if 'custom_eps' in kwargs:
        for agent in agents:
            agent.model.set_epsilon(kwargs.get('custom_eps'))

    # -------------------------------------- #
    # region: Main game loop                 #
    # -------------------------------------- #
    for epoch in range(cfg.experiment.epochs):


        # Reset the environment and all agents
        env.reset()
        agents = env.agents
        for agent in agents:
            agent.reset()

        # Reset the turn variables
        done, turn, points = 0, 0, 0

        # ---------------------------------- #
        # region: Main turn loop             #
        # ---------------------------------- #
        while done == 0:

            # Increment the turn counter
            turn += 1

            for agent in agents:

                # Actions to take before the agent acts
                agent.model.start_epoch_action(**locals())

                # Agent transitions
                state, action, reward, next_state, done_ = agent.transition(env)
                
                # Done if the agent finds a truck or the max turns have been reached.
                if (turn >= cfg.experiment.max_turns) or done_:
                    done = True

                # Add rewards to the total game points
                points += reward

                # Add the experience to the agent's memory
                exp = (1, (state, action, reward, next_state, done))
                agent.episode_memory.append(
                    exp
                )

                # Actions to take after the agent acts.
                agent.model.end_epoch_action(**locals())

        # ---------------------------------- #
        # endregion: Main turn loop          #
        # ---------------------------------- #

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:
            loss = agent.model.train_model()
            
            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, loss.detach().numpy(), points)

        # Print the variables to the console
        game_vars.pretty_print()

        # Add scalars to Tensorboard
        if cfg.log:
            writer.add_scalar('Loss', loss, epoch)
            writer.add_scalar('Reward', points, epoch)
            writer.add_scalars(
                'Encounters',
                {
                    'Korean': agents[0].encounters['korean'],
                    'Lebanese': agents[0].encounters['lebanese'],
                    'Mexican': agents[0].encounters['mexican'],
                    'Wall': agents[0].encounters['wall']
                }, epoch)
    # -------------------------------------- #
    # endregion: Main game loop              #
    # -------------------------------------- #

    # Close the tensorboard log
    if cfg.log:
        writer.close()
    
    # If a file path has been specified, save the weights to the specified path
    if 'save_weights' in kwargs:
        for agent in agents:
            agent.model.save(file_path = kwargs.get('save_weights'))

if __name__ == '__main__':

    # Load the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=f'{root}/examples/ft/config.yaml', help='The filepath to YAML configuration file.')
    args = parser.parse_args()
    cfg = load_config(args)

    # Set up the run parameters (including where to save and load the models)
    model_directory = f'{root}/examples/ft/checkpoints'
    run_params = [
        # {
        #     'custom_eps': 0.7,
        #     'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.7}.pkl'
        # },
        # {
        #     'load_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.7}.pkl',
        #     'custom_eps': 0.5,
        #     'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.5}.pkl'
        # },
        # {
        #     'load_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.5}.pkl',
        #     'custom_eps': 0.3,
        #     'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.3}.pkl'
        # },
        # {
        #     'load_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.3}.pkl',
        #     'custom_eps': 0.1,
        #     'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.1}.pkl'
        # },
        # {
        #     'load_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.1}.pkl',
        #     'custom_eps': 0.0,
        #     'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.0}.pkl'
        # },
        {
            'load_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.0}_4.pkl',
            'custom_eps': 0.0,
            'save_weights': f'{model_directory}/model_{cfg.model.iqn.type}_{0.0}_5.pkl'
        }
    ]

    # Run each parameterization of the model.
    for i in range(len(run_params)):

        run_game(
            cfg,
            **run_params[i]
            )
