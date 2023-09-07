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
from examples.food_trucks.models.worldmodel import (
    PrioritizedReplayMemory,
    sample_and_visualize_test
)
from examples.RPG3_PPO.PPO_CPC import (
    RolloutBuffer,
    PPO
)

# from examples.food_trucks.models.cpc import CPCModel
from examples.food_trucks.models.worldmodel import WorldModel

# # # # # # # # # # # #
# Setup trained model #
# # # # # # # # # # # #

def setup(multiplier = 1000):

    world_size = 11
    memory_size = 5
    vision = 5
    env = FoodTrucks(
        height=world_size,
        width=world_size,
        layers=1,
        truck_prefs=(10,5,-5),
        baker_mode=True,
        one_hot = True,
        vision = vision,
        full_mdp=True
    )

    turn = 1
    trainable_models = [0]

    # Set up model and environment
    models = create_models(n_agents = len(trainable_models),
                           memory_size = memory_size,
                           vision = vision)

    # Set up parameters (epsilon, epochs, max_turns)
    run_params = (
        [0.5, 5 * multiplier],
        [0.1, 10 * multiplier],
        [0.0, 10 * multiplier],
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
            memory_size = memory_size,
            log = False
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
            "output_padding": 2,  # Add this line
        },
        {"layer_type": "relu"},
        {
            "layer_type": "conv",
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "output_padding": 2,  # Add this line
        },
        # Note, max pool is broken in the config
    ]
    
    models = []
    for i in range(n_agents):
        models.append(
            WorldModel(
                input_shape=[7, 11, 11],
                cpc_output = 64,
                action_space = 4,
                batch_size=1024,
                cnn_config = cnn_config,
                memories = stored_memories,
                memory_size=5,
                no_cnn=True,
                device = 'cpu'
            )
        )

    return models

def run_model(models, device, plot = False, plot_freq = 100, n_epochs = 100000, action_model = None, env = None):
    for model in models:
        for epoch in range(n_epochs):
            cpc, r1, r2 = model.train()
            if plot:
                if epoch % plot_freq == 0:
                    clear_output(wait = True)
                    print(f'Epoch: {epoch}. CPC loss: {round(cpc, 2)}. R1 loss: {round(r1, 2)}. R2 loss: {round(r2, 2)}.')
                    sample_and_visualize_test(model, 
                                              device, 
                                              one_hot = True, 
                                              epoch = epoch,
                                              action_model=action_model,
                                              env=env)


