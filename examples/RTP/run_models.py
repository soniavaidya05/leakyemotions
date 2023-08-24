# ---------------------------- #
#     Fix import structure     #
# ---------------------------- #
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
module_path = os.path.abspath("../../..")
if module_path not in sys.path:
    sys.path.append(module_path)

from examples.RTP.env import RTP
from examples.RTP.main import run_game, create_models

# ---------------------------- #
#    Run game test with viz    #
# ---------------------------- #

parameters = {
    "world_size": 20,  # Size of the environment
    "num_models": 1,  # Number of agents. Right now, only supports 1
    "sync_freq": 200,  # Parameters related to model soft update. TODO: Figure out if these are still needed
    "model_update_freq": 4,  # Parameters related to model soft update. TODO: Figure out if these are still needed
    "epsilon": 0.3,  # Exploration parameter
    "conditions": [
        "None",
        "implicit_attitude",
        "EWA",
        "implicit_attitude+EWA",
    ],  # Model run conditions
    "epsilon_decay": 0.999,  # Exploration decay rate
    "episodic_decay_rate": 1.0,  # EWA episodic decay rate
    "similarity_decay_rate": 1.0,
    "epochs": 2000,  # Number of epochs
    "max_turns": 20,  # Number of turns per game
    "object_memory_size": 2500,  # Size of the memory buffer
    "knn_size": 10,  # Size of the nearest neighbours
    "RUN_PROFILING": False,  # Whether to time each epoch
    "log": False,  # Tensorboard support. Currently disabled
    "contextual": False,  # Whether the agents' need changes based on its current resource value or stays static
    "appearance_size": 20,
}

# Run model with all of the conditions
for condition in range(len(parameters["conditions"])):
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
