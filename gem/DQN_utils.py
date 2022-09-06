import numpy as np
from models.perception import agentVisualField
import torch
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from game_utils import createWorld, createWorldImage
import matplotlib.animation as animation

import pickle


def createVideo(models, worldSize, num, env, filename="unnamed_video.gif"):
    fig = plt.figure()
    ims = []
    env.reset_env(worldSize, worldSize)
    gamePoints = [0, 0]
    for _ in range(num):
        image = createWorldImage(env.world)
        im = plt.imshow(image, animated=True)
        ims.append([im])
        gamePoints = env.step(models, gamePoints)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


def save_models(models, env, save_dir, filename, add_videos):
    with open(save_dir + filename, "wb") as fp:
        pickle.dump(models, fp)
    for video_num in range(add_videos):
        vfilename = save_dir + filename + "_replayVid_" + str(video_num) + ".gif"
        createVideo(models, 17, video_num, env, vfilename)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model
