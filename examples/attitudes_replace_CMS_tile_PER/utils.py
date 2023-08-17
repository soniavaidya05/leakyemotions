import numpy as np
from gem.models.perception import agent_visualfield
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def find_moveables(world):
    # see if the last places this is being used this is needed
    moveList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].static == 0:
                    moveList.append((i, j, k))
    return moveList


def find_agents(world):
    # update gems and wolves to use find_instance instead of this and remove this code
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].kind == "agent":
                    agentList.append((i, j, k))
    return agentList


def find_instance(world, kind):
    # needs to be rewriien to return location (i, j, k)
    instList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].action_type == kind:
                    instList.append((i, j, k))
    return instList


def update_epsilon(epsilon, rate=0.99):
    # TODO, this goes down way too fast when turn is low
    # so, fast updatee mean that epsilon goes down faster than it should
    # if epsilon > 0.1:
    #    epsilon -= 1 / (turn)

    epsilon = max(0.1, rate * epsilon)
    return epsilon


def examine_replay(models, index, modelnum):
    image_r = models[modelnum].replay[index][0][0][0].numpy()
    image_g = models[modelnum].replay[index][0][0][1].numpy()
    image_b = models[modelnum].replay[index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay[index][3][0][0].numpy()
    image_g = models[modelnum].replay[index][3][0][1].numpy()
    image_b = models[modelnum].replay[index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay[index][1]
    reward = models[modelnum].replay[index][2]
    done = models[modelnum].replay[index][4]

    return image, image2, (action, reward, done)


# look at a few replay games
def replay_games(numGames, modelNum, startingMem, models):
    for i in range(numGames):
        image, image2, memInfo = examine_replay(models, i + startingMem, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")

        print(memInfo)

        plt.show()


def examine_replay_memory(models, episode, index, modelnum):
    image_r = models[modelnum].replay.memory[episode][index][0][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][0][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay.memory[episode][index][3][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][3][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay.memory[episode][index][1]
    reward = models[modelnum].replay.memory[episode][index][2]
    done = models[modelnum].replay.memory[episode][index][4]

    return image, image2, (action, reward, done)


# look at a few replay games


def replay_MemoryGames(models, modelNum, episode):
    epLength = len(models[modelNum].replay.memory[episode])
    for i in range(epLength):
        image, image2, memInfo = examine_replay_memory(models, episode, i, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")
        print(memInfo)
        plt.show()


def number_memories(modelNum, models):
    episodes = len(models[modelNum].replay.memory)
    print("there are ", episodes, " in the model replay buffer.")
    for e in range(episodes):
        epLength = len(models[0].replay.memory[e])
        print("Memory ", e, " is ", epLength, " long.")


def update_memories(env, expList, done, end_update=True):
    # update the reward and last state after all have moved
    # changed to holdObject to see if this fixes the failure of updating last memory
    for loc in expList:
        # location = (i, j, 0)
        # holdObject = env.world[loc]
        exp = env.world[loc].episode_memory[-1]
        lastdone = exp[1][4]
        if done == 1:
            lastdone = 1
        if end_update == False:
            exp = exp[0], (
                exp[1][0],
                exp[1][1],
                env.world[loc].reward,
                exp[1][3],
                lastdone,
            )
        if end_update == True:
            input2 = env.pov(loc)
            exp = exp[0], (
                exp[1][0],
                exp[1][1],
                env.world[loc].reward,
                input2,
                lastdone,
            )
        env.world[loc].episode_memory[-1] = exp
    return env.world


def transfer_world_memories(models, world, expList, extra_reward=True):
    # transfer the events from agent memory to model replay
    for loc in expList:
        # this moves the specific form of the replay memory into the model class
        # where it can be setup exactly for the model
        models[world[loc].policy].transfer_memories(world, loc, extra_reward=True)
    return models


def update_memories_rnn(env, expList, done, end_update=True):
    # update the reward and last state after all have moved
    # changed to holdObject to see if this fixes the failure of updating last memory
    for loc in expList:
        # location = (i, j, 0)
        # holdObject = env.world[loc]
        exp = env.world[loc].episode_memory[-1]
        lastdone = exp[1][4]
        if done == 1:
            lastdone = 1
        if end_update == False:
            exp = exp[0], (
                exp[1][0],
                exp[1][1],
                env.world[loc].reward,
                exp[1][3],
                lastdone,
                exp[1][5],
                exp[1][6],
            )
        if end_update == True:
            input2 = env.pov(loc)
            exp = exp[0], (
                exp[1][0],
                exp[1][1],
                env.world[loc].reward,
                input2,
                lastdone,
                exp[1][5],
                exp[1][6],
            )
        env.world[loc].episode_memory[-1] = exp
    return env.world


def plot_time_decay(input_size, time_decay_rate=1):
    N = input_size
    time_weights_linear = np.arange(N) / (N - 1)
    time_weights_exp = np.exp(-np.arange(N) / (N - 1) * time_decay_rate)

    plt.plot(time_weights_linear, label="Linear Decay")
    plt.plot(
        time_weights_exp, label="Exponential Decay (rate={})".format(time_decay_rate)
    )
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Time Decay Comparison")
    plt.show()


# Example usage
# plot_time_decay(100, time_decay_rate=2)

from examples.attitudes_replace_CMS_tile_PER.elements import EmptyObject, Wall


def make_Q_map(env, models, value_model, world_size, sparce=0.01, save_name=None):
    Q_array1 = np.zeros((world_size, world_size))
    R_array1 = np.zeros((world_size, world_size))
    QR_array1 = np.zeros((world_size, world_size))

    Q_array2 = np.zeros((world_size, world_size))
    R_array2 = np.zeros((world_size, world_size))
    QR_array2 = np.zeros((world_size, world_size))

    Q_array3 = np.zeros((world_size, world_size))
    R_array3 = np.zeros((world_size, world_size))
    QR_array3 = np.zeros((world_size, world_size))

    env.reset_env(
        height=world_size,
        width=world_size,
        layers=1,
        gem1p=sparce,
        gem2p=sparce,
        gem3p=sparce,
        change=False,
    )
    env.change_gem_values()

    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(1)
        env.world[loc].init_rnn_state = None

    agentList = find_instance(env.world, "neural_network")
    agent = env.world[agentList[0]]
    env.world[agentList[0]] = EmptyObject()

    # pass 1

    for i in range(world_size - 2):
        for j in range(world_size - 2):
            loc = (i + 1, j + 1, 0)
            original_content = env.world[
                loc
            ]  # Save what was originally at the location
            locReward = original_content.value
            R_array1[i + 1, j + 1] = locReward

            env.world[loc] = agent  # Put agent in place
            state = env.pov(loc)
            Qs = models[0].qnetwork_local.get_qvalues(state)
            Q = torch.max(Qs).detach().item()
            Q_array1[i + 1, j + 1] = Q
            QR_array1[i + 1, j + 1] = Q + locReward

            env.world[
                loc
            ] = original_content  # Put back what was originally at the location

    # pass 2

    # for i in range(world_size):
    #    for j in range(world_size):
    #        object_state = torch.tensor(env.world[i, j, 0].appearance[:3]).float()
    #        rs, _ = value_model(object_state.unsqueeze(0))
    #        r = rs[0][1]
    #        env.world[i, j, 0].appearance[3] = r.item() * 255

    for i in range(world_size - 2):
        for j in range(world_size - 2):
            loc = (i + 1, j + 1, 0)
            original_content = env.world[
                loc
            ]  # Save what was originally at the location
            locReward = original_content.value
            R_array2[i + 1, j + 1] = locReward
            env.world[i + 1, j + 1, 0].appearance[3] = locReward * 255
            # env.world[i + 1, j + 1, 0].appearance[4] = locReward * 255

            env.world[loc] = agent  # Put agent in place
            state = env.pov(loc)
            Qs = models[0].qnetwork_local.get_qvalues(state)
            Q = torch.max(Qs).detach().item()
            Q_array2[i + 1, j + 1] = Q
            QR_array2[i + 1, j + 1] = Q + locReward

            env.world[
                loc
            ] = original_content  # Put back what was originally at the location

    # pass 3

    # for i in range(world_size):
    #    for j in range(world_size):
    #        object_state = torch.tensor(env.world[i, j, 0].appearance[:3]).float()
    #        rs, _ = value_model(object_state.unsqueeze(0))
    #        r = rs[0][1]
    #        r = (r * -1) + 5
    #        env.world[i, j, 0].appearance[3] = r.item() * 255
    #        env.world[i, j, 0].appearance[4] = r.item() * 255

    for i in range(world_size - 2):
        for j in range(world_size - 2):
            loc = (i + 1, j + 1, 0)
            original_content = env.world[
                loc
            ]  # Save what was originally at the location
            locReward = original_content.value
            R_array3[i + 1, j + 1] = locReward
            locReward = (locReward * -1) + 5
            env.world[i + 1, j + 1, 0].appearance[3] = locReward * 255
            # env.world[i + 1, j + 1, 0].appearance[4] = locReward * 255

            env.world[loc] = agent  # Put agent in place
            state = env.pov(loc)
            Qs = models[0].qnetwork_local.get_qvalues(state)
            Q = torch.max(Qs).detach().item()
            Q_array3[i + 1, j + 1] = Q
            QR_array3[i + 1, j + 1] = Q + locReward

            env.world[
                loc
            ] = original_content  # Put back what was originally at the location

    plt.subplot(3, 3, 1)  # First subplot
    plt.imshow(Q_array1, cmap="viridis")  # Plot the first array
    plt.colorbar()  # To add a color scale
    plt.title("IQN Q", fontsize=8)  # Title for the first plot

    plt.subplot(3, 3, 2)  # Second subplot
    plt.imshow(R_array1, cmap="viridis")  # Plot the second array
    plt.colorbar()  # To add a color scale
    plt.title("R", fontsize=8)  # Title for the second plot

    plt.subplot(3, 3, 3)  # Third subplot
    plt.imshow(QR_array1, cmap="viridis")  # Plot the second array
    plt.colorbar()  # To add a color scale
    plt.title("IQN QR", fontsize=8)  # Title for the third plot

    plt.subplot(3, 3, 4)  # First subplot
    plt.imshow(Q_array2, cmap="viridis")  # Plot the first array
    plt.colorbar()  # To add a color scale
    plt.title("IQN + implicit Q", fontsize=8)  # Title for the first plot

    plt.subplot(3, 3, 5)  # Second subplot
    plt.imshow(R_array2, cmap="viridis")  # Plot the second array
    plt.colorbar()  # To add a color scale
    plt.title("R", fontsize=8)  # Title for the second plot

    plt.subplot(3, 3, 6)  # Third subplot
    plt.imshow(QR_array2, cmap="viridis")  # Plot the second array
    plt.colorbar()  # To add a color scale
    plt.title("IQN + implicit QR", fontsize=8)  # Title for the first plot

    plt.subplot(3, 3, 7)  # First subplot
    plt.imshow(Q_array3, cmap="viridis")  # Plot the first array
    plt.colorbar()  # To add a color scale
    plt.title("IQN + implicit (flipped)", fontsize=8)  # Title for the first plot

    plt.subplot(3, 3, 8)  # Second subplot

    # Reshape the arrays for scatterplot
    x_data = Q_array2.ravel()
    y_data = Q_array3.ravel()

    # Filter out points where either x or y is zero
    mask = (x_data != 0) & (y_data != 0)
    x_data = x_data[mask]
    y_data = y_data[mask]
    # Calculate the correlation coefficient
    corr_coeff = np.corrcoef(x_data, y_data)[0, 1]

    sns.regplot(x=x_data, y=y_data, scatter_kws={"s": 5}, line_kws={"color": "red"})
    plt.title(f"Correlation: Q_array2 vs Q_array3\nr = {corr_coeff:.2f}", fontsize=8)

    plt.subplot(3, 3, 9)  # Third subplot
    plt.imshow(QR_array3, cmap="viridis")  # Plot the second array
    plt.colorbar()  # To add a color scale
    plt.title("IQN + implicit QR (flipped)", fontsize=8)  # Title for the first plot

    if save_name:  # If save_name is provided, save the figure to a file
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free up memory
    else:  # If save_name is not provided, show the plot on the screen
        plt.show()
