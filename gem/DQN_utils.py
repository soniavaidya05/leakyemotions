from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from game_utils import createWorld, createWorldImage
import matplotlib.animation as animation

import pickle

from gem.utils import (
    updateMemories,
    findMoveables,
)


def createVideo(models, worldSize, num, env, filename="unnamed_video.gif"):
    fig = plt.figure()
    ims = []
    env.reset_env(worldSize, worldSize)
    done = 0
    for i, j in findMoveables(env.world):
        # reset the memories for all agents
        env.world[i, j, 0].init_replay(3)
    gamePoints = [0, 0]
    for _ in range(num):
        image = createWorldImage(env.world)
        im = plt.imshow(image, animated=True)
        ims.append([im])
        gamePoints = env.step(models, gamePoints, 0.1)
        env.world = updateMemories(
            models, env.world, findMoveables(env.world), done, endUpdate=False
        )

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


def save_models(models, save_dir, filename):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def makeVideo(filename, save_dir, models, worldSize, env):
    epoch = 10000
    for video_num in range(5):
        vfilename = (
            save_dir
            + filename
            + "_replayVid_"
            + str(epoch)
            + "_"
            + str(video_num)
            + ".gif"
        )
        createVideo(models, worldSize, 100, env, filename=vfilename)


def replayView(memoryNum, agentNumber, env):
    agentList = findMoveables(env.world)
    i, j = agentList[agentNumber]

    Obj = env.world[i, j, 0]

    state = Obj.replay[memoryNum][0]
    next_state = Obj.replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()


def replayViewModel(memoryNum, modelNumber, models):
    state = models[modelNumber].replay[memoryNum][0]
    next_state = models[modelNumber].replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()
    print(
        models[modelNumber].replay[memoryNum][1],
        models[modelNumber].replay[memoryNum][2],
        models[modelNumber].replay[memoryNum][4],
    )


def createData(env, models, epochs, worldSize):
    gamePoints = [0, 0]
    env.reset_env(worldSize, worldSize)
    for i, j in findMoveables(env.world):
        env.world[i, j, 0].init_replay(3)
    for _ in range(epochs):
        gamePoints = env.step(models, gamePoints)
    return env
