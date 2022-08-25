#!/usr/bin/env python
# coding: utf-8

# from gemGame import runCombinedTraining, moreTraining
from gemGame_experimental_LSTM import (
    train_wolf_gem,
    save_models,
    load_models,
    train_wolf_gem,
    addTrain_wolf_gem,
)

# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

models = train_wolf_gem(10000)
save_models(models, save_dir, "lstm_build_replay_memories_10000", 10)

models = addTrain_wolf_gem(models, epochs=10000, epsilon=0.65)
save_models(models, save_dir, "lstm_build_replay_memories_20000", 10)

models = addTrain_wolf_gem(models, epochs=10000, epsilon=0.54)
save_models(models, save_dir, "lstm_build_replay_memories_30000", 10)

models = addTrain_wolf_gem(models, epochs=10000, epsilon=0.44)
save_models(models, save_dir, "lstm_build_replay_memories_40000", 10)

models = addTrain_wolf_gem(models, epochs=10000, epsilon=0.2)
save_models(models, save_dir, "lstm_build_replay_memories_50000", 10)
