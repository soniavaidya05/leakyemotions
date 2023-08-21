import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

# ------------------------------------- #
# region: Generic gem helper functions  #
# ------------------------------------- #

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

def update_terminal_memories(models, env, done, end_update=True):

    agent_locations = find_instance(env.world, "neural_network")

    '''Combine above two functions'''
    env.world = update_memories(env, agent_locations, done, end_update=end_update)
    models = transfer_world_memories(models, env.world, agent_locations)
    return models, env

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

def initialize_rnn(env, num_memories):
    '''
    Reset the memories for all agents in the environment
    
    env: the environment
    num_memories: the length of the sequence of empty memories to initialize
    '''
    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        env.world[loc].init_replay(num_memories)
        env.world[loc].init_rnn_state = None

# endregion

# ------------------------------------- #
# region: RTP-specific helper functions #
# ------------------------------------- #

def viz(env, show = True):

    '''
    Visualize the environment state using an RGB color map.
    show (Optional): A flag determining whether to show the environment (true by default), or return the Matplotlib image
    '''

    # Set up a color map corresponding to objects in the environment
    color_map = {
        'agent': [200.0, 0.0, 0.0], # agent is red
        'gemwood': [0.0, 200.0, 0.0], # mostly-choppers are green
        'gemstone': [150.0, 150.0, 150.0], # mostly-miners are grey
        'empty': [0.0, 0.0, 0.0],
        'wall': [50.0, 50.0, 50.0]
    }

    # Three channels for each pixel
    image = np.zeros((env.height, env.width, 3))

    # Traverse through each pixel
    for i in range(env.height):
        for j in range(env.width):
            # Get the element name
            pixel = env.world[i, j, 0].kind
            # For gems, append wood or stone based on their skill
            if pixel == 'gem':
                resource = 'wood' if env.world[i, j, 0].appearance[3] == 255.0 else 'stone'
                pixel += resource
            # Set the pixel value to the colormap value for the element
            image[i, j, :] = color_map[pixel]
    
    # Show the image
    if show:
        plt.imshow(image.astype(int))
        plt.show()
    # Or return a PIL Image
    else:
        plt.imshow(image.astype(int))
        import io
        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

def create_frame(env):
    
    '''
    Create a video frame from an environment
    Alias for viz(env, show = False)
    '''
    return viz(env, show = False)

def assemble_frames(imgs, filename, folder = '/Users/rgelpi/Documents/GitHub/transformers/examples/RTP/data/'):
    
    '''
    Take an array of frames and assemble them into a GIF with the given path.
    imgs: the array of frames
    filename: A filename to save the images to
    folder: The path to save the gif to
    run_num: (Optional) allows saving multiple images in a single run
    '''
    path = folder + filename + '.gif'

    imgs[0].save(path, format = 'GIF', append_images = imgs[1:], save_all = True, duration = 100, loop = 0)

def run_one_game(
        all_models,
        condition,
        env,
        max_turns=100,
        save = False,
        run_id = 0,
        view = True
    ):
    '''
    Run a single game.
    all_models: A tuple including (IQN model, Value model, Resource model, EWA model). If you are only using one of these, the rest passed in can be None
    '''

    # Unpack the models
    models, value_models, resource_models, ewa_models = all_models
    if value_models is not None:
        value_model = value_models[0]
    if resource_models is not None:
        resource_model = resource_models[0]
    if ewa_models is not None:
        ewa_model = ewa_models[0]

    done, turn = 0, 0
    total_reward = 0
    env.reset_env()

    # Initialize the RNN
    initialize_rnn(env, num_memories = 1)

    # Create the array and append the first image to it
    if save:
        imgs = []
        img = create_frame(env)
        imgs.append(img)
    # View the first frame
    if view:
        print('Initial state.')
        viz(env)


    while done == 0:
        
        turn+=1

        # Get the location of agents
        agent_locations = find_instance(env.world, "neural_network")
        random.shuffle(agent_locations)

        for loc in agent_locations:

            # Renaming "holdObject" to "agent" for readability
            agent = env.world[loc]
            # Reset turn reward to 0
            agent.reward = 0

            # Observation of the environment at agent's location
            state = env.pov(loc)

            # Act according to model policy
            action = models[agent.policy].take_action(state, 0.0)

            # Get the environment transitions
            (
                env.world,
                reward,
                next_state,
                done,
                new_loc,
                object_info,
                resource_outcome,
            ) = agent.transition(env, models, action[0], loc)

            # Add frames to image
            if save:
                img = create_frame(env)
                imgs.append(img)
            # View the next frame
            if view:
                clear_output(wait = True)
                movement = ['up', 'down', 'left', 'right']
                print(f'Took action: {movement[action[0]]}. Received {reward}.')
                viz(env)

            # Create object state
            state_object = object_info[0:-3]

            # ---------------------------------------- #
            # Implicit attitude: train the value model #
            # ---------------------------------------- #
                            
            if 'implicit' in condition:
                value_model.add_memory(state_object, reward)

                # When the replay buffer is long enough, begin training the model
                if len(value_model.replay_buffer) > 51 and turn % 2 == 0:
                    memories = value_model.sample(50)
                    value_loss = value_model.learn(memories, 25)

            # ---------------------------------------- #
            # Resource guess: train the resource model #
            # ---------------------------------------- #

            if 'tree_rocks' in condition:

                # learn resource of target
                if reward != 0:
                    resource_model.add_memory(state_object, resource_outcome)
                else:
                    if random.random() > 0.5:  # seems to work if downsample nothing
                        resource_model.add_memory(state_object, resource_outcome)

                # When the replay buffer is long enough, begin training the model
                if len(resource_model.replay_buffer) > 33 and turn % 2 == 0:
                    resource_loss = resource_model.learn(
                        resource_model.sample(32), batch_size=32
                    )

            # ---------------------------------------- #
            # EWA model: Add episodic memory and train #
            # ---------------------------------------- #

            if 'EWA' in condition:

                # Add the state-reward pair to the episodic memory buffer
                ewa_model.memory.append((state_object, reward))
                ewa_model.fit()


            # End the game when the last turn is reached
            if (turn > max_turns):
                done = 1
                print(f'Game over. Received {total_reward} total reward.')

            # Add experience to replay
            exp = (
                1, # Priority value
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                ),
            )

            # Update agent episode memory and total reward
            agent = env.world[new_loc]
            agent.episode_memory.append(exp)
            total_reward += reward
    
    if save:
        # Save the file
        assemble_frames(imgs,
                        filename=f'game_{condition}_{run_id}')

# endregion