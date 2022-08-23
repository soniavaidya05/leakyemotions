#!/usr/bin/env python
# coding: utf-8

# from gemGame import runCombinedTraining, moreTraining
from gemGame_experimental_LSTM import (
    train_wolf_gem,
    save_models,
    load_models,
    playGameTest,
    playGame,
    train_wolf_gem_LSTM5,
)
from models.cnn_lstm_dqn import model_CNN_LSTM_DQN

# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

models = train_wolf_gem_LSTM5(50000)
save_models(models, save_dir, "lstm_build_replay_memories", 3)


models = []
models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 650, 350, 100, 4))  # agent model
models.append(model_CNN_LSTM_DQN(5, 0.0001, 1500, 2570, 350, 100, 4))  # wolf model
models, world = playGameTest(
    models,  # model file list
    [0, 1],  # which models from that list should be trained, here not the agents
    15,  # world size
    1000,  # number of epochs
    100,  # max epoch length
    0.85,  # starting epsilon
    gameVersion="wolvesGems",  # which game to play
)


from gem.utils import (
    findInstance,
    one_hot,
    updateEpsilon,
    updateMemories,
    transferMemories,
    findMoveables,
    findAgents,
)

import torch


moveList = findMoveables(world)
print(moveList)

i, j = moveList[0]

t1 = world[i, j, 0].replay[-5][0]
t2 = world[i, j, 0].replay[-4][0]
t3 = world[i, j, 0].replay[-3][0]
t4 = world[i, j, 0].replay[-2][0]
t5 = world[i, j, 0].replay[-1][0]

seq = torch.cat([t1, t2, t3, t4, t5], dim=1)


def transferMemories_LSTM(models, world, expList, extraReward=True):
    # transfer the events from agent memory to model replay
    for i, j in expList:

        # note, should these replay[0]s be replay[-1] in case we need to store more memories?
        exp = world[i, j, 0].replay[-1]

        t1 = world[i, j, 0].replay[-5][0]
        t2 = world[i, j, 0].replay[-4][0]
        t3 = world[i, j, 0].replay[-3][0]
        t4 = world[i, j, 0].replay[-2][0]
        t5 = world[i, j, 0].replay[-1][0]

        seq1 = torch.cat([t1, t2, t3, t4], dim=1)
        seq2 = torch.cat([t2, t3, t4, t5], dim=1)

        exp[0] = seq1
        exp[3] = seq2

        models[world[i, j, 0].policy].replay.append(exp)
        if extraReward == True and abs(exp[2]) > 9:
            for _ in range(5):
                models[world[i, j, 0].policy].replay.append(exp)
    return models
