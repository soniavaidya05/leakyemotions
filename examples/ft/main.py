# --------------- #
# region: Imports #
import os
import sys
module_path = os.path.abspath('../..')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from examples.ft.config import (
    init_log,
    parse_args,
    load_config,
    create_models
)
from examples.ft.utils import add_models
from examples.ft.env import FoodTrucks

import random
import numpy as np
# endregion       #
# --------------- #


def run(cfg):

    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    # Create the environment
    env = FoodTrucks(cfg)

    # Create the models and add them to the agents
    models = create_models(cfg)
    agents = env.get_entities('Agent')
    add_models(agents, models)

    for epoch in range(cfg.experiment.epochs):

        # Reset the environment
        env.reset()
        for agent in agents:
            agent.reset()
        random.shuffle(agents)
        
        done = 0
        turn = 0
        points = 0
        losses = 0

        while not done:

            turn += 1
            if turn >= cfg.experiment.max_turns:
                done = 1

            for agent in agents:
                agent.model.start_epoch_action(**locals())

            # Agent transition
            for agent in agents:

                (state,
                action,
                reward,
                next_state,
                ) = agent.transition(env)

                agent.encounter(reward)

                exp = (0, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                points += reward

                # Update world memories
                agent.model.end_epoch_action(**locals())

        # Train each agent after an epoch
        for agent in agents:
            """
            Train the neural networks at the end of eac epoch, reduced to 64 so that the new memories ~200 are slowly added with the priority ones
            """
            loss = agent.model.training()
            agent.episode_memory.clear() 
            losses = losses + loss.detach().cpu().numpy()

            # Special action: update epsilon
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)

            

        if cfg.log:
            writer.add_scalar('Reward', points, epoch)
            writer.add_scalar('Loss', losses, epoch)
            writer.add_scalar('Korean', np.sum([agent.encounters['korean'] for agent in agents]), epoch)
            writer.add_scalar('Lebanese', np.sum([agent.encounters['lebanese'] for agent in agents]), epoch)
            writer.add_scalar('Mexican', np.sum([agent.encounters['mexican'] for agent in agents]), epoch)
            writer.add_scalar('Wall', np.sum([agent.encounters['wall'] for agent in agents]), epoch)        
        else:
            print(f'''
Epoch {epoch}:
--------------
Loss: {round(losses, 2)}
Points: {round(points, 2)}
Epsilon {round(agent.model.epsilon, 4)}
--------------
                  ''')
        
def main():
    args = parse_args()
    cfg = load_config(args)
    init_log(cfg)
    run(cfg)

if __name__ == '__main__':
    main()