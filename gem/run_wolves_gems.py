from gem.DQN_utils import make_video, save_models

from gem.gemsWolves_LSTM_DQN import run_game


save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

# needs a dictionary with the following keys:
# turn, trainable_models, sync_freq, modelUpdate_freq

# below needs to be written
# env, epsilon, params = setup_game(world_size=15)


run_params = (
    [0.9, 1000, 5],
    [0.8, 5000, 5],
    [0.7, 5000, 5],
    [0.2, 5000, 5],
    [0.8, 10000, 25],
    [0.6, 10000, 35],
    [0.2, 10000, 35],
    [0.2, 20000, 50],
)

# the version below needs to have the keys from above in it
for modRun in range(len(run_params)):
    models, env, turn, epsilon = run_game(
        models,
        env,
        turn,
        run_params[modRun][0],
        epochs=run_params[modRun][1],
        max_turns=run_params[modRun][2],
    )
    save_models(models, save_dir, "newWolvesAndAgents" + str(modRun))


make_video("test_new", save_dir, models, 20, env)
