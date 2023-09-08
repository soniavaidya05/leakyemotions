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
from examples.ft.config import create_models, load_config
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

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to config file")
args = parser.parse_args(['--config', '../ft/config.yaml'])
cfg = load_config(
    args
)

def run_game(
    models,
    env: FoodTrucks,
    epochs,
    max_turns,
    sync_freq,
    model_update_freq,
    epsilon
):
    
    losses = []
    points = []

    for agent, model in zip(agents, models):
        agent.model = model
        agent.model.epsilon = epsilon

    for epoch in range(epochs):

        env.reset()
        agents = env.get_entities(
            'Agent'
        )
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

                state, action, reward, next_state = agent.transition(env)

                points.append(reward)
                
                agent.encounter(reward)

                exp = (1, (state, action, reward, next_state, done))

                agent.episode_memory.append(
                    exp
                )

                agent.models.transfer_memories(
                    agent, extra_reward = True
                )

                if epoch > 200 and turn % model_update_freq == 0:
                    loss = agent.model.training()
                    losses.append(loss)
        
        if epoch > 100:

            for agent in agents:
                loss = agent.model.training()
                losses.append(loss)

        if len(losses) > 10:

            print(f'''
--------------------------------------------------------------------
Epoch: {epoch} | Turns: {turn} | Running mean loss: {round(np.mean(losses[-10:]), 2)} | Running mean reward: {round(np.mean(points[-10:]), 2)}
''')
        else:
            print(f'''
--------------------------------------------------------------------
Epoch: {epoch} | Turns: {turn} | Running mean loss: [n/a] | Running mean reward: {round(np.mean(points[-10:]), 2)}
''')
        
    return models

env = FoodTrucks(cfg)
models = create_models(cfg)

run_params = {
    'models': models,
    'env': env,
    'epochs': 5000,
    'max_turns': 100,
    'sync_freq': 200,
    'model_update_freq': 4,
    'epsilon': 0.5
}

eps = [0.5, 0.1, 0.01]
for i in range(3):
    run_params['epsilon'] = eps[i]
    models = run_game(
        **run_params
    )
    


            






                    

            
        
