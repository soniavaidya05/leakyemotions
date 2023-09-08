# --------------- #
# region: Imports #
import os
import sys
module_path = os.path.abspath('../..')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from examples.ft.models.iqn import iRainbowModel
from examples.ft.env import FoodTrucks
from examples.ft.entities import EmptyObject
from examples.ft.utils import visual_field
from examples.ft.config import create_models, create_agents, create_entities, load_config
from examples.food_trucks.main import run_game

import torch
import numpy as np
import random
import argparse

# endregion       #
# --------------- #

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def run_game(
    models,
    env: FoodTrucks,
    epochs,
    max_turns,
    sync_freq,
    model_update_freq,
    epsilon,
    log
):
    if log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    losses = []
    points = []

    for epoch in range(epochs):

        env.reset()
        agents = env.agents
        for agent in agents:
            agent.reset()

        done, turn = 0, 0

        while done == 0:

            turn += 1

            if turn >= max_turns:
                done = 1

            for agent in agents:

                if epoch % sync_freq == 0:
                    agent.model.qnetwork_target.load_state_dict(
                        agent.model.qnetwork_local.state_dict()
                    )

                state, action, reward, next_state, done_ = agent.transition(env)
                
                if done_:
                    done = True

                points.append(reward)

                exp = (1, (state, action, reward, next_state, done))

                agent.episode_memory.append(
                    exp
                )

                agent.model.transfer_memories(
                    agent, extra_reward = True
                )

                if epoch > 200 and turn % model_update_freq == 0:
                    loss = agent.model.training()
                    losses.append(loss.detach().numpy())
        
        if epoch > 100:

            for agent in agents:
                loss = agent.model.training()
                losses.append(loss.detach().numpy())

        if len(losses) > 0:

            print(f'''
--------------------------------------------------------------------
Epoch: {epoch} | Turns: {turn} | Running mean loss: {losses[-1]} | Running mean reward: {np.mean(points)}
''')
            if log:
                writer.add_scalar('Loss', losses[-1], epoch)
                writer.add_scalar('Reward', points[-1], epoch)
                writer.add_scalars('Encounters',
                                   {
                                       'Korean': agents[0].encounters['korean'],
                                       'Lebanese': agents[0].encounters['lebanese'],
                                       'Mexican': agents[0].encounters['mexican'],
                                       'Wall': agents[0].encounters['wall']
                                   }, epoch)
        else:
            print(f'''
--------------------------------------------------------------------
Epoch: {epoch} | Turns: {turn} | Running mean loss: [n/a] | Running mean reward: {np.mean(points)}
''')
    
    if log:
        writer.close()
        
    return models

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', default='../ft/config.yaml', help='The filepath to YAML configuration file.')
args = parser.parse_args()
cfg = load_config(args)

# Create model objects and environment
models = create_models(cfg)
agents = create_agents(cfg, models)
entities = create_entities(cfg)
env = FoodTrucks(cfg, agents, entities)

run_params = {
    'models': models,
    'env': env,
    'epochs': cfg.experiment.epochs,
    'max_turns': cfg.experiment.max_turns,
    'sync_freq': cfg.model.iqn.parameters.sync_freq,
    'model_update_freq': 4,
    'epsilon': 0.5,
    'log': cfg.log
}

# Epsilon values to run
eps = [0.5, 0.1, 0.01]

for i in range(3):
    run_params['epsilon'] = eps[i]
    models = run_game(
        **run_params
    )

    for model in models:
        torch.save(
            {
                'local': model.qnetwork_local.state_dict(),
                'target': model.qnetwork_target.state_dict(),
                'optim': model.optimizer.state_dict()
            },
            f'./data/model_{eps[i]}.pkl'
        )


            






                    

            
        
