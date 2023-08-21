# ------------------------------------------ #
#                                            #
# Main file for running the RTP environment. #
#                                            #
# If this file is called, run the game with  #
# the parameter configuration dictionary.    #
#                                            #
# NOTE: Proper run syntax for this file is:  #
#                                            #
# python main.py > /path/to/output_file.txt  #
#                                            #
# ------------------------------------------ #

parameters = {
    'world_size': 25, # Size of the environment
    'num_models': 1, # Number of agents. Right now, only supports 1
    'sync_freq': 200, # Parameters related to model soft update. TODO: Figure out if these are still needed
    'model_update_freq': 4, # Parameters related to model soft update. TODO: Figure out if these are still needed 
    'epsilon': 0.5, # Exploration parameter
    'conditions': ['EWA', 'None'], # Model run conditions
    'epsilon_decay': 0.9999, # Exploration decay rate
    'episodic_decay_rate': 1.0, # EWA episodic decay rate
    'similarity_decay_rate': 1.0, # EWA similarity decay rate
    'epochs': 1000, # Number of epochs
    'max_turns': 20, # Number of turns per game
    'object_memory_size': 12000, # Size of the memory buffer
    'knn_size': 5, # Size of the nearest neighbours
    'RUN_PROFILING': False, # Whether to time each epoch
    'log': False, # Tensorboard support. Currently disabled
    'contextual': True # Whether the agents' need changes based on its current resource value or stays static
}

#region: Imports

# ---------------------------- #
#     Fix import structure     #
# ---------------------------- #

'''
Note from Rebekah: This is weirdness that only affects my computer, 
and can be commented out as long as you don't have any issues with
package imports on your existing gem installation.
'''
fix_imports = True
if fix_imports:
    print(f'Fixing import structure...')
    import os
    import sys
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    module_path = os.path.abspath('../..')
    if module_path not in sys.path:
        sys.path.append(module_path)

# ---------------------------- #
# Import RTP-specific packages #
# ---------------------------- #
from examples.RTP.utils import (
    update_terminal_memories,
    find_instance,
    initialize_rnn
)
from examples.RTP.models.iRainbow_clean import iRainbowModel
from examples.RTP.models.attitude_models import (
    ValueModel, 
    ResourceModel, 
    EWAModel,
    evaluate
)
from examples.RTP.env import RTP

# ------------------------------
#      Additional packages     #
# ---------------------------- #
import torch
import time
import random

# Seed information
SEED = time.time() 
random.seed(SEED)
torch.manual_seed(SEED)

#endregion: Imports

def create_models(num_models = 1, device = 'cpu', **kwargs):
    """
    Should make the sequence length of the LSTM part of the model and an input here
    Should also set up so that the number of hidden laters can be added to dynamically
    in this function. Below should fully set up the NN in a flexible way for the studies
    """

    models = []
    value_models = []
    resource_models = []
    ewa_models = []
    for i in range(num_models):
        models.append(
            iRainbowModel(
                in_channels=5,
                num_filters=5,
                cnn_out_size=567,  # 910
                state_size=torch.tensor(
                    [5, 9, 9]
                ),  # this seems to only be reading the first value
                action_size=4,
                layer_size=250,  # 100
                n_step=3,  # Multistep IQN (rainbow paper uses 3)
                BATCH_SIZE=64,
                BUFFER_SIZE=1024,
                LR=0.00025,  # 0.00025
                TAU=1e-3,  # Soft update parameter
                GAMMA=0.95,  # Discout factor 0.99
                N=12,  # Number of quantiles
                device=device,
                seed=SEED,
            )
        )
        value_models.append(
            ValueModel(
                state_dim = 8,
                memory_size=250,
            )
        )
        resource_models.append(
            ResourceModel(
                state_dim = 8,
                memory_size = 2000
            )
        )

        ewa_models.append(
            EWAModel(
                mem_len=250,
                state_knn_len=5,
                episodic_decay_rate=kwargs['episodic_decay_rate'],
                similarity_decay_rate=kwargs['similarity_decay_rate']
            )
        )

    return models, value_models, resource_models, ewa_models

def run_game(
    all_models,
    env,
    condition,
    parameters
):
    
    '''
    Unpack models and parameters
    '''
    models, value_models, resource_models, ewa_models = all_models
    # For now, just use this by default
    value_model = value_models[0]
    resource_model = resource_models[0]
    ewa_model = ewa_models[0]
    epochs = parameters['epochs']
    sync_freq = parameters['sync_freq']
    model_update_freq = parameters['model_update_freq']
    log = parameters['log']
    epsilon = parameters['epsilon']
    contextual = env.contextual

    # Tensorboard logging. Currently disabled because meltingpot broke my tensorboard installation :(
    if log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    
    """
    This is the main loop of the game
    """

    losses = 0
    total_reward = 0
    approaches = {
        'rewarding': 0,
        'unrewarding': 0,
        'wall': 0
    }
    change = True

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """

        # Reduce epsilon by decay value each epoch
        epsilon = epsilon * parameters['epsilon_decay']
        # Reset done flag and turn counter
        done, turn = 0, 0

        # Reset the environment
        env.reset_env(

            change=change, # Change the values if the change flag is set to True
        )

        # Reset agent memories with the specified number of memories
        working_memory = 1
        initialize_rnn(env, working_memory)


        # -------------- #
        # Attitude model #
        # -------------- #

        # If the learning is not contextual, it only needs to be learned once per epoch
        if not contextual:
            # Call the attitude model evaluation before each epoch
            if epoch > 10:
                evaluate(
                    env,
                    condition=condition,
                    value_model=value_model,
                    resource_model=resource_model,
                    ewa_model=ewa_model
                )


        start_time = time.time()

        while done == 0:
            """
            Run through each turn
            """
            turn += 1


            # --------------------------------------------------------------
            # note the sync models lines may need to be deleted
            # the IQN has a soft update, so we should test dropping
            # the lines below

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for model in models:
                    model.qnetwork_target.load_state_dict(
                        model.qnetwork_local.state_dict()
                    )
            # --------------------------------------------------------------


            # Get the location of agents
            agent_locations = find_instance(env.world, "neural_network")
            random.shuffle(agent_locations)

            for loc in agent_locations:

                # -------------------------------------- #
                # Contextual attitudes: update each turn #
                # -------------------------------------- #

                if contextual:
                    # Reset the appearance of objects at the given location
                    env.reset_appearance(loc)

                    if epoch > 10:
                        evaluate(
                            env,
                            condition=condition,
                            value_model=value_model,
                            resource_model=resource_model,
                            ewa_model=ewa_model,
                            loc = loc
                        )


                # Renaming "holdObject" to "agent" for readability
                agent = env.world[loc]
                # Reset turn reward to 0
                agent.reward = 0

                # Observation of the environment at agent's location
                state = env.pov(loc)

                # Act according to model policy
                action = models[agent.policy].take_action(state, epsilon)

                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                    object_appearance,
                    resource_outcome,
                ) = agent.transition(env, models, action[0], loc)

                # Record reward outcomes

                # Rewarding: Agent chose the right resource
                # Unrewarding: Agent chose the wrong resource (e.g. wood when it needed stone)
                # Wall: Agent ran into a wall
                if reward == 10:
                    approaches['rewarding'] += 1
                elif reward == 0 and resource_outcome in [[0, 1, 0] or [0, 0, 1]]:
                    approaches['unrewarding'] += 1
                elif reward == -1:
                    approaches['wall'] += 1

                # Create object state from the first component of the object appearance
                state_object = object_appearance[0:-3]

                # ---------------------------------------- #
                # Implicit attitude: train the value model #
                # ---------------------------------------- #
                                
                if 'implicit' in condition:
                    value_model.add_memory(state_object, reward)

                    # When the replay buffer is long enough, begin training the model
                    if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                        memories = value_model.sample(50)
                        value_loss = value_model.learn(memories, 25)

                # ---------------------------------------- #
                # Resource guess: train the resource model #
                # ---------------------------------------- #

                if 'tree_rocks' in condition:

                    # learn resource of target
                    if reward != 0:
                        resource_model.add_memory(state_object, resource_outcome)
                    else:
                        if random.random() > 0.5:  # seems to work if downsample nothing
                            resource_model.add_memory(state_object, resource_outcome)

                    # When the replay buffer is long enough, begin training the model
                    if len(resource_model.replay_buffer) > 33 and turn % 2 == 0:
                        resource_loss = resource_model.learn(
                            resource_model.sample(32), batch_size=32
                        )

                # ---------------------------------------- #
                # EWA model: Add episodic memory and train #
                # ---------------------------------------- #

                if 'EWA' in condition:

                    # Add the state-reward pair to the episodic memory buffer
                    ewa_model.memory.append((state_object, reward))
                    ewa_model.fit()


                # End the game when the last turn is reached
                if (turn > parameters['max_turns']):
                    done = 1

                # Add experience to replay
                exp = (
                    1, # Priority value
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    ),
                )

                # Update agent episode memory and total reward
                agent = env.world[new_loc]
                agent.episode_memory.append(exp)
                total_reward += reward

            # Transfer memories after the episode is complete
            models, env = update_terminal_memories(models, env, done)

            # Update the neural networks within an episode at a rate of model_update_freq
            # Not sure why this is repeated below
            # TODO: Test whether both of these are needed?
            if epoch > 10 and turn % model_update_freq == 0:

                for model in models:
                    experiences = model.memory.sample()
                    loss = model.learn(experiences)
                    losses = losses + loss

        # At the end of each epoch, train the neural networks
        if epoch > 10:
            for model in models:
                experiences = model.memory.sample()
                loss = model.learn(experiences)
                losses = losses + loss

        end_time = time.time()
        if parameters['RUN_PROFILING']:
            print(f"Epoch {epoch} took {end_time - start_time} seconds")

        # Every 20 epochs, log the results and reset the counters
        if epoch % 20 == 0 and len(models) > 0 and epoch != 0:
            # Tensorboard scalar logging
            if log:
                writer.add_scalar('Reward', total_reward)
                writer.add_scalar('Successful Approaches', approaches['rewarding'])
                writer.add_scalar('Unsuccessful Approaches', approaches['unrewarding'])
                writer.add_scalar('Collisions', approaches['wall'])
                writer.add_scalar('Losses', losses)
                
            # Old print statement for supporting previous results plot framework
            print(
                epoch,
                turn,
                round(total_reward),
                [approaches['rewarding'], approaches['unrewarding'], approaches['wall']],
                losses,
                epsilon,
                str(0),
                condition,
            )
            total_reward = 0
            approaches = {
                'rewarding': 0,
                'unrewarding': 0,
                'wall': 0
            }
            losses = 0
    
    # Repack all models when the model is done
    all_models = (models, value_models, resource_models, ewa_models)
    return all_models, env, turn, epsilon

if __name__ == '__main__':

    env = RTP(
        height=parameters['world_size'],
        width=parameters['world_size'],
        layers=1,
        contextual=parameters['contextual']
    )

    for condition in range(len(parameters['conditions'])):
        
        # Create new models
        all_models = create_models(
            num_models = 1,
            device = 'cpu',
            decay_rate = parameters['episodic_decay_rate'] # kwarg included for EWA model
        )

        # Run game with a new set of models
        all_models, env, turn, epsilon = run_game(
            all_models = all_models,
            env = env,
            condition = parameters['conditions'][condition],
            parameters = parameters
        )
