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
    "world_size": 25,  # Size of the environment
    "num_models": 1,  # Number of agents. Right now, only supports 1
    "sync_freq": 200,  # Parameters related to model soft update. TODO: Figure out if these are still needed
    "model_update_freq": 4,  # Parameters related to model soft update. TODO: Figure out if these are still needed
    "epsilon": 0.3,  # Exploration parameter
    "conditions": ["None", "EWA"],  # Model run conditions
    "epsilon_decay": 0.999,  # Exploration decay rate
    "episodic_decay_rate": 1.0,  # EWA episodic decay rate
    "similarity_decay_rate": 1.0,  # EWA similarity decay rate
    "epochs": 1000,  # Number of epochs
    "max_turns": 20,  # Number of turns per game
    "object_memory_size": 12000,  # Size of the memory buffer
    "knn_size": 5,  # Size of the nearest neighbours
    "RUN_PROFILING": False,  # Whether to time each epoch
    "log": False,  # Tensorboard support. Currently disabled
    "contextual": True,  # Whether the agents' need changes based on its current resource value or stays static
    "appearance_size": 20,
}

# ---------------------------- #
# region: imports              #
# ---------------------------- #

# ---------------------------- #
#     Fix import structure     #
# ---------------------------- #

"""
Note from Rebekah: This is weirdness that only affects my computer, 
and can be commented out as long as you don't have any issues with
package imports on your existing gem installation.
"""
fix_imports = True
if fix_imports:
    print(f"Fixing import structure...")
    import os
    import sys

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    module_path = os.path.abspath("../..")
    if module_path not in sys.path:
        sys.path.append(module_path)

# ---------------------------- #
# Import RTP-specific packages #
# ---------------------------- #
from examples.RTP.utils import update_terminal_memories, find_instance, initialize_rnn
from examples.RTP.models.iRainbow_clean import iRainbowModel
from examples.RTP.models.attitude_models import (
    ValueModel,
    ResourceModel,
    EWAModel,
    evaluate,
)
from examples.RTP.env import RTP

# ------------------------------
#      Additional packages     #
# ---------------------------- #
import torch
import time
import random
import numpy as np

# Seed information
SEED = time.time()
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------- #
# endregion: imports           #
# ---------------------------- #


def create_models(num_models=1, device="cpu", **kwargs):
    """
    Create N models for the RTP game.
    Currently, only 1 is supported for all attitude models
    """

    models = []
    value_models = []
    resource_models = []
    ewa_models = []
    for i in range(num_models):
        models.append(
            iRainbowModel(
                state_size=torch.tensor(
                    [kwargs["appearance_size"], 9, 9]
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
                state_dim=kwargs["appearance_size"] - 3,
                memory_size=250,
            )
        )
        resource_models.append(
            ResourceModel(state_dim=kwargs["appearance_size"] - 3, memory_size=2000)
        )

        ewa_models.append(
            EWAModel(
                mem_len=2500,
                state_knn_len=kwargs["knn_size"],
                episodic_decay_rate=kwargs["episodic_decay_rate"],
                similarity_decay_rate=kwargs["similarity_decay_rate"],
            )
        )

    return models, value_models, resource_models, ewa_models


def run_game(
    all_models,
    env,
    epsilon,
    epochs=10000,
    max_turns=100,
    epsilon_decay=0.999,
    condition="implicit_attitude",
    sync_freq=200,
    model_update_freq=4,
    RUN_PROFILING=False,
):
    # Unpack the models
    models, value_models, resource_models, ewa_models = all_models
    # For now, just use 1 attitude model
    value_model = value_models[0]
    resource_model = resource_models[0]
    ewa_model = ewa_models[0]
    # Whether the environmental rewards are contextual
    contextual = env.contextual

    # Set up metrics
    losses = 0
    total_reward = 0
    outcomes = [0, 0, 0, 0]
    approaches = {"rewarding": 0, "unrewarding": 0, "wall": 0}

    for epoch in range(epochs):
        """
        Move each agent once and then update the world
        Creates new gamepoints, resets agents, and runs one episode
        """
        epsilon = epsilon * epsilon_decay
        done, turn = 0, 0

        env.reset_env()

        working_memory = 1
        for loc in find_instance(env.world, "neural_network"):
            # reset the memories for all agents
            # the parameter sets the length of the sequence for LSTM
            env.world[loc].init_replay(working_memory)
            env.world[loc].init_rnn_state = None

            # Set starting resource for agents
            start_resource = np.random.choice([0, 1])
            if start_resource == 0:
                env.world[loc].wood = 1
                env.world[loc].stome = 0
            if start_resource == 1:
                env.world[loc].wood = 0
                env.world[loc].stome = 1

        # If the environment is static, evaluate attitude models each turn
        if not contextual:
            evaluate(env, condition, value_model, resource_model, ewa_model, epoch)

        # Start time for logging
        start_time = time.time()

        while done == 0:
            """
            Find the agents and move them
            """
            turn = turn + 1

            # -------------------------------------------------------------- #
            # region: sync_freq
            # note the sync models lines may need to be deleted
            # the IQN has a soft update, so we should test dropping
            # the lines below

            if epoch % sync_freq == 0:
                # update the double DQN model ever sync_frew
                for model in models:
                    model.qnetwork_target.load_state_dict(
                        model.qnetwork_local.state_dict()
                    )

            # endregion
            # -------------------------------------------------------------- #

            agent_locations = find_instance(env.world, "neural_network")

            random.shuffle(agent_locations)

            # -------------------------------------------------------------- #
            # region: each agent action loop...                              #
            # -------------------------------------------------------------- #
            for loc in agent_locations:
                # Renaming "holdObject" to "agent" for readability
                agent = env.world[loc]
                # Reset turn reward to 0
                agent.reward = 0

                # -------------------------------------- #
                # Contextual attitudes: update each turn #
                # -------------------------------------- #

                if contextual:
                    # Reset environment appearance
                    env.reset_appearance(loc)

                    # Update attitude models
                    evaluate(
                        env,
                        condition=condition,
                        value_model=value_model,
                        resource_model=resource_model,
                        ewa_model=ewa_model,
                        epoch=epoch,
                        loc=loc,
                    )

                # Observation of the environment at agent's location
                state = env.pov(loc)

                # Act according to model policy
                action = models[agent.policy].take_action(state, epsilon)

                # Agent transitions
                (
                    env.world,
                    reward,
                    next_state,
                    done,
                    new_loc,
                    object_appearance,
                    resource_outcome,
                    outcome_type,
                ) = agent.transition(env, models, action[0], loc)
                if outcome_type >= 0:
                    outcomes[outcome_type] += 1

                # Create object state from the first component of the object appearance
                state_object = object_appearance[0:-3]

                # ---------------------------------------- #
                # region: update the attitude models...    #

                # ---------------------------------------- #
                # Implicit attitude: train the value model #
                # ---------------------------------------- #

                if "implicit" in condition:
                    value_model.add_memory(state_object, reward)

                    # When the replay buffer is long enough, begin training the model
                    if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                        memories = value_model.sample(50)
                        value_loss = value_model.learn(memories, 25)

                # ---------------------------------------- #
                # Resource guess: train the resource model #
                # ---------------------------------------- #

                if "tree_rocks" in condition:
                    # learn resource of target
                    if reward != 0:
                        resource_model.add_memory(state_object, resource_outcome)
                    else:
                        if random.random() > 0.9:  # seems to work if downsample nothing
                            resource_model.add_memory(state_object, resource_outcome)

                    # When the replay buffer is long enough, begin training the model
                    if len(resource_model.replay_buffer) > 33:  # and turn % 2 == 0:
                        resource_loss = resource_model.learn(
                            resource_model.sample(32), batch_size=32
                        )

                # ---------------------------------------- #
                # EWA model: Add episodic memory and train #
                # ---------------------------------------- #

                if "EWA" in condition:
                    # Add the state-reward pair to the episodic memory buffer
                    ewa_model.memory.append((state_object, reward))
                    ewa_model.fit()

                # endregion: update the attitude models... #
                # ---------------------------------------- #

                # Record reward outcomes

                # Rewarding: Agent chose the right resource
                # Unrewarding: Agent chose the wrong resource (e.g. wood when it needed stone)
                # Wall: Agent ran into a wall
                if reward == 10:
                    approaches["rewarding"] += 1
                # elif reward == 0 and 255.0 in object_appearance[2:4]:
                #    approaches['unrewarding'] += 1
                elif reward == -6:
                    approaches["unrewarding"] += 1
                elif reward == -1:
                    approaches["wall"] += 1

                if turn > max_turns:
                    done = 1

                exp = (
                    1,  # Priority value
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    ),
                )

                env.world[new_loc].episode_memory.append(exp)
                total_reward += reward

            # -------------------------------------------------------------- #
            # endregion: each agent action loop                              #
            # -------------------------------------------------------------- #

            # Transfer memories after the episode is complete
            models, env = update_terminal_memories(models, env, done)

            # Update the neural networks within an episode at a rate of model_update_freq
            # Not sure why this is repeated below
            # TODO: Test whether both of these are needed?
            if epoch > 10 and turn % model_update_freq == 0:
                """
                Train the neural networks within a eposide at rate of modelUpdate_freq
                """
                for model in models:
                    experiences = model.memory.sample()
                    loss = model.learn(experiences)
                    losses = losses + loss

        # At the end of each epoch, train the neural networks
        if epoch > 10:
            for model in models:
                """
                Train the neural networks at the end of eac epoch
                reduced to 64 so that the new memories ~200 are slowly added with the priority ones
                """
                experiences = model.memory.sample()
                loss = model.learn(experiences)
                losses = losses + loss

        # Time length of each epoch
        end_time = time.time()
        if RUN_PROFILING:
            print(f"Epoch {epoch} took {end_time - start_time} seconds")

        if epoch % 20 == 0 and epoch != 0:
            # Old print statement for supporting previous results plot framework
            print(
                epoch,
                turn,
                round(total_reward),
                *[
                    approaches["rewarding"],
                    approaches["unrewarding"],
                    approaches["wall"],
                ],
                losses,
                *outcomes,
                epsilon,
                str(0),
                condition,
            )
            # rs = show_weighted_averaged(object_memory)
            # print(epoch, rs)
            total_reward = 0
            approaches = {"rewarding": 0, "unrewarding": 0, "wall": 0}
            losses = 0
            outcomes = [0, 0, 0, 0]

    # Repack all models when the model is done
    all_models = models, value_models, resource_models, ewa_models
    return all_models, env


if __name__ == "__main__":
    # the version below needs to have the keys from above in it
    for condition in range(len(parameters["conditions"])):
        # Create new models
        all_models = create_models(
            appearance_size=parameters["appearance_size"],
            episodic_decay_rate=parameters["episodic_decay_rate"],
            similarity_decay_rate=parameters["similarity_decay_rate"],
            knn_size=parameters["knn_size"],
        )

        env = RTP(
            height=parameters["world_size"],
            width=parameters["world_size"],
            layers=1,
            contextual=parameters["contextual"],
        )

        all_models, env = run_game(
            all_models,
            env,
            epsilon=parameters["epsilon"],
            epochs=parameters["epochs"],
            max_turns=parameters["max_turns"],
            epsilon_decay=parameters["epsilon_decay"],
            condition=parameters["conditions"][condition],
            sync_freq=parameters["sync_freq"],
            model_update_freq=parameters["model_update_freq"],
            RUN_PROFILING=parameters["RUN_PROFILING"],
        )
