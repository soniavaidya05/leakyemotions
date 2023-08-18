# Import basic packages
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output
import json
import argparse

from examples.food_trucks.main import run_game, create_models
from examples.food_trucks.env import FoodTrucks
from examples.food_trucks.utils import generate_memories
from examples.food_trucks.worldmodel import (
    PrioritizedReplayMemory,
    WorldModel,
    sample_and_visualize_test
)
from examples.RPG3_PPO.PPO_CPC import (
    RolloutBuffer,
    PPO
)

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# # # # # # # # # # # #
# Setup trained model #
# # # # # # # # # # # #

def setup(multiplier = 1000):

    world_size = 11
    env = FoodTrucks(
        height=world_size,
        width=world_size,
        layers=1,
        truck_prefs=(10,5,-5),
        baker_mode=True,
        one_hot = True
    )

    turn = 1
    trainable_models = [0]

    # Set up model and environment
    models = create_models(n_agents = len(trainable_models))

    # Set up parameters (epsilon, epochs, max_turns)
    run_params = (
        [0.5, multiplier],
        [0.1, 2 * multiplier],
        [0.0, 3 * multiplier],
    )

    # Train model
    for modRun in range(len(run_params)):
        models, env, turn, epsilon = run_game(
            models,
            env,
            turn,
            epsilon = run_params[modRun][0],
            epochs=run_params[modRun][1],
            max_turns=100,
            world_size=world_size,
            trainable_models = trainable_models,
            sync_freq = 200,
            modelUpdate_freq = 4,
            log = True
        )

    # Generate memories
    memories = generate_memories(
        models,
        env,
        n_games=10 * multiplier
    )



    return models, env, memories

def create_replay(n_agents,
                  one_hot,
                  memories):
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    # Store them in a replay buffer
    n_mem = len(memories)
    stored_memories = PrioritizedReplayMemory(capacity = n_mem)
    for i in range(n_mem):
        stored_memories.push(memories[i])
        if i % 100 == 0:
            clear_output(wait = True)
            print(f'Storing memory {i+1}...')

    if one_hot:
        n_channels = 7
    else:
        n_channels = 3

    cnn_config = [
        {
            "layer_type": "conv",
            "in_channels": n_channels,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 3,
            "padding": 0,
        },
        {"layer_type": "relu"},
        {
            "layer_type": "conv",
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            # "output_padding": 1,  # Add this line
        },
        # Note, max pool is broken in the config
    ]
    
    models = []
    for i in range(n_agents):
        models.append(
            WorldModel(
                capacity = len(stored_memories),
                input_shape = (n_channels, 9, 9),
                output_shape = (n_channels, 9, 9),
                num_actions = 4,
                batch_size = 1024,
                cnn_config = cnn_config,
                stored_memories = stored_memories
            )
        )

    return models

def run_model(models, device, plot = False, plot_freq = 100, n_epochs = 100000, action_model = None, env = None):
    writer = SummaryWriter()
    for model in models:
        losses1 = 0
        losses2 = 0
        for epoch in range(n_epochs):
            reconstruction_loss1, reconstruction_loss2 = model.train_model(device)
            losses1 = losses1 + reconstruction_loss1
            losses2 = losses2 + reconstruction_loss2
            writer.add_scalar("Losses 1", reconstruction_loss1, epoch)
            writer.add_scalar("Losses 2", reconstruction_loss2, epoch)
            if plot:
                if epoch % plot_freq == 0:
                    clear_output(wait = True)
                    print(f'Epoch: {epoch}. R1 loss: {round(reconstruction_loss1, 2)}. R2 loss: {round(reconstruction_loss2, 2)}.')
                    sample_and_visualize_test(model, 
                                              device, 
                                              one_hot = True, 
                                              epoch = epoch,
                                              action_model=action_model,
                                              env=env)
    writer.close()



