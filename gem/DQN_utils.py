from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from game_utils import create_world, create_world_image
import matplotlib.animation as animation

import pickle

from gem.utils import (
    update_memories,
    find_moveables,
)


def create_video(models, world_size, num, env, filename="unnamed_video.gif"):
    fig = plt.figure()
    ims = []
    env.reset_env(world_size, world_size)
    done = 0
    for location in find_moveables(env.world):
        # reset the memories for all agents
        env.world[location].init_replay(3)
    game_points = [0, 0]
    for _ in range(num):
        image = create_world_image(env.world)
        im = plt.imshow(image, animated=True)
        ims.append([im])
        game_points = env.step(models, game_points, 0.1)
        env.world = update_memories(
            models, env.world, find_moveables(env.world), done, end_update=False
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


def make_video(filename, save_dir, models, world_size, env):
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
        create_video(models, world_size, 100, env, filename=vfilename)


def replay_view(memoryNum, agentNumber, env):
    agentList = find_moveables(env.world)
    location = agentList[agentNumber]

    Obj = env.world[location]

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


def replay_view_model(memoryNum, modelNumber, models):
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


def create_data(env, models, epochs, world_size):
    game_points = [0, 0]
    env.reset_env(world_size, world_size)
    for i, j in find_moveables(env.world):
        env.world[i, j, 0].init_replay(3)
    for _ in range(epochs):
        game_points = env.step(models, game_points)
    return env
