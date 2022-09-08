import numpy as np
from models.perception import agentVisualField
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from game_utils import createWorld, createWorldImage
import matplotlib.animation as animation

import pickle

from gem.utils import findMoveables


def createVideo(models, worldSize, num, env, filename="unnamed_video.gif"):
    fig = plt.figure()
    ims = []
    env.reset_env(worldSize, worldSize)
    for i, j in findMoveables(env.world):
        # reset the memories for all agents
        env.world[i, j, 0].init_replay(3)
    gamePoints = [0, 0]
    for _ in range(num):
        image = createWorldImage(env.world)
        im = plt.imshow(image, animated=True)
        ims.append([im])
        done = 0
        envStepVer = 0
        if envStepVer == 1:
            gamePoints = env.step(models, gamePoints)

        if envStepVer == 0:

            moveList = findMoveables(env.world)
            for i, j in moveList:
                env.world[i, j, 0].reward = 0

            for i, j in moveList:

                holdObject = env.world[i, j, 0]

                if holdObject.static != 1:
                    input = models[holdObject.policy].pov(
                        env.world, i, j, holdObject
                    )
                    action = models[holdObject.policy].takeAction([input, .1])

                if holdObject.has_transitions == True:
                    env.world, models, gamePoints = holdObject.transition(
                        action,
                        env.world,
                        models,
                        i,
                        j,
                        gamePoints,
                        done,
                        input,
                    )


        gamePoints = env.step(models, gamePoints)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model
