# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import argparse
import random

from IPython.display import clear_output
from matplotlib import pyplot as plt
from typing import Union, Sequence
from datetime import datetime

# Import gem-specific packages
from gem.utils import (
    visual_field, 
    visual_field_sprite,
    animate,
    random_seed, 
    set_seed, 
    fig2img, 
    nearest_2_power, 
    minmax,
)
from gem.models.buffer import GameReplayBuffer as Buffer
from gem.models.transformer import VisionTransformer as ViT

# Import RPG-specific packages
from examples.RPG.env import RPG
from examples.RPG.agents import color_map
from examples.RPG.utils import (
    load_config,
    create_agents,
    create_entities,
    create_models,
    Cfg
)
from examples.trucks.utils import GameVars

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
    env = RPG(cfg, agents, entities)

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
            gamma=cfg.model.iqn.parameters.GAMMA,
            timesteps=100,
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
            img = visual_field_sprite(env.world, tile_size = env.tile_size)
            clear_output(wait = True)
            print(f'╔═════════════════════════════════════════════════════╗')
            print(f'║                 Initial game state...               ║')
            print(f'╚═════════════════════════════════════════════════════╝')
            plt.imshow(img.astype(int))
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

            entities = env.get_entities_for_transition()
            # Entity transition
            for entity in entities:
                entity.transition(env)

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
                    img = visual_field_sprite(env.world, tile_size=env.tile_size)
                    clear_output(wait = True)
                    scores.pretty_print(*flags, epoch=game, turn=turn, reward=points)
                    plt.imshow(img.astype(int))
                    plt.title(f'Game {seed}. Turn {turn}. Action: {ACTIONS[action]}. Reward: {reward}')
                    frame = fig2img(plt)
                    frames.append(frame)
                    plt.show()
                    
        
        # ----------------- #
        # endregion         #
        # ----------------- #

        # Collect information for the scores and the frames, and reset the batched action deque.
        if 'memories' in flags:
            buffer.game.clear()
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

def train_transformer_model(cfg: Cfg, action_model_pattern):

    # ------------------------------------------------------ #
    # region: Create the forward model for the animations... #
    # ------------------------------------------------------ #

    models = create_models(cfg)
    for model in models:
        model.load(f'{cfg.root}/examples/RPG/models/checkpoints/{action_model_pattern}.pkl')
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    # env = RPG(cfg, agents, entities)
    memories = Buffer.load(f'{cfg.root}/examples/RPG/data/{action_model_pattern}_memories.pkl')
    memories.batch_size = 64 # Reduce batch size since the # of frames is quite long
    # action_model_store = Buffer(
    #     buffer_size=cfg.experiment.max_turns,
    #     batch_size=cfg.experiment.max_turns,
    #     device=None,
    #     seed=None,
    #     gamma=None,
    #     timesteps=cfg.experiment.max_turns,
    # )

    # ------------------------------------------------------ #
    # endregion                                              #
    # ------------------------------------------------------ #

    inverse_model = ViT(
        state_size=cfg.model.iqn.parameters.state_size,
        action_space=4,
        layer_size=200,
        patch_size=3,
        num_frames=99,
        num_heads=4,
        batch_size=64,
        num_layers=4,
        memory=memories,
        LR=.0001,
        device='cpu',
        seed=random.randint(0,1000)
    )

    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            log_dir=f'{cfg.root}/examples/RPG/runs/inverse/{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )
    
    for epoch in range(10000):
        state_loss, action_loss = inverse_model.train_model()
        state_predictions, state_targets = inverse_model.plot_trajectory()

        if cfg.log:
            writer.add_scalar('Action loss', action_loss, epoch)
            writer.add_scalar('State loss', state_loss, epoch)
            writer.add_images('State targets', state_targets[:5], epoch) # First 5 images
            writer.add_images('State preds', state_predictions[:5], epoch)

        # if epoch % 50 == 0:
        #     clear_output(wait = True)

        #     # ----------------- #
        #     # region: Game loop #
        #     # ----------------- #

        #     env.reset()
        #     for agent in agents:
        #         agent.reset()
        #         agent.model.epsilon = 0
        #         # Evaluation mode
        #         agent.model.eval()

        #     done, turn = 0, 0

        #     while done == 0:

        #         # ----------------- #
        #         # region: Turn loop #
        #         # ----------------- #

        #         for agent in agents:

        #             state, action, reward, next_state, done_ = agent.transition(env)

        #             # If agent finished or max turns is reached, end the turn
        #             if turn >= cfg.experiment.max_turns or done_: 
        #                 done = 1 # End loop if maximum turns have been reached
                    
        #             # Append the experience to the replay
        #             exp = (0, (state, action, reward, next_state, done))
        #             agent.episode_memory.append(exp)
        #             action_model_store.add(state, action, reward, next_state, done)

        #         # ----------------- #
        #         # endregion         #
        #         # ----------------- #

        #     frames = []

        #     while len(action_model_store.memory) > 0:

        #         experience = action_model_store.memory.popleft()
        #         state, action, _, next_state, _ = experience

        #         clear_output(wait = True)
        #         fig = inverse_model.plot_images(state, action.squeeze(), next_state)
        #         img = fig2img(fig)
        #         plt.show()
        #         frames.append(img)

        #     animate(frames,
        #             f'transformer_model_{epoch}.png',
        #             '../data/')
                        
        #     # ----------------- #
        #     # endregion         #
        #     # ----------------- #        


        