# --------------- #
# region: Imports #
# --------------- #

# Import base packages
from typing import Optional
from numpy.typing import ArrayLike 
import torch

# Import gem packages
from gem.utils import visual_field
from gem.models.ann import ANN
from gem.primitives import Object, GridworldEnv
from examples.trucks.utils import color_map

# --------------- #
# endregion       #
# --------------- #

# ----------------------------------------------------- #
# region: Memory class                                  #

class Memory:
    '''
    Memory for the agent class
    '''

    def __init__(self, memory_size: int):
        self.priorities = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextstates = []
        self.dones = []

    def clear(self):
        del self.priorities[:]
        del self.actions[:]
        del self.states[:]
        del self.nextstates[:]
        del self.rewards[:]
        del self.dones[:]

    def append(self, exp: tuple):
        '''
        Add an experience to the agent's memory.
        
        Parameters:
            exp: The tuple to add 
        '''
        # Unpack
        priority, exp1 = exp
        state, action, reward, nextstate, done = exp1
        # Add to replay
        self.priorities.append(priority)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.nextstates.append(nextstate)
        self.dones.append(done)

    def get_last_memory(self, attr: Optional[str] = None):
        '''
        Get the latest memory from the replay.

        Parameters:
            attr: (Optional) the attribute to get.
            If not specified, it returns all elements
        '''
        if attr is None:
            return (
                self.priorities[-1], 
                self.states[-1], 
                self.actions[-1], 
                self.rewards[-1], 
                self.nextstates[-1], 
                self.dones[-1]
            )
        else:
            return getattr(self, attr)[-1]

# endregion
# ----------------------------------------------------- #

# ----------------------------------------------------- #
# region: Agent class for the Baker ToM task            #
# ----------------------------------------------------- #

class Agent(Object):
    '''
    Base agent object.

    Parameters:

        color: The appearance of the agent (list of N floats) \n
        model: The model object \n
        memory_size: The size of the replay buffer
    '''
    def __init__(self, 
                 appearance: list, 
                 model: ANN, 
                 cfg,
                 location = None
                ):
        super().__init__(appearance)
        self.cfg = cfg
        self.vision = self.cfg.vision
        self.location = location
        self.has_transitions = True

        # Training-related features
        self.episode_memory = Memory(self.cfg.memory_size)
        self.model = model

        self.encounters = {
            'korean': 0,
            'lebanese': 0,
            'mexican': 0,
            'wall': 0
        }


    def init_replay(self,  
                    state_shape: Optional[ArrayLike] = None
                    ) -> None:
        '''
        Fill in blank images for the LSTM. Requires the state size to be fully defined.

        Parameters:

            state_shape: (Optional) a tuple or list of C x H x W of the state size.
            If it is not specified, the state size will be specified with the agent's vision.
        '''
        priority = torch.tensor(0.1)
        num_frames = self.model.num_frames

        if state_shape is not None:
            state = torch.zeros(1, num_frames, *state_shape).float()
        else:
            C = len(self.appearance)
            H = W = self.vision * 2 + 1
            state = torch.zeros(1, num_frames, C, H, W).float()

        
        action = torch.tensor(4.0) # Action outside the action space
        reward = torch.tensor(0.0)
        done = torch.tensor(0.0)

        # Priority, (state, action, reward, nextstate, done)
        exp = (priority, (state, action, reward, state, done))

        self.episode_memory.append(exp)
        
    def movement(self,
                 action: int
                 ) -> tuple:
        
        '''
        Takes an action and returns a new location
        '''
        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        return new_location
    
    def pov(self,
            env: GridworldEnv) -> torch.Tensor:
        '''
        Defines the agent's observation function
        '''

        # Get the previous state
        previous_state = self.episode_memory.get_last_memory('states')

        # Get the frames from the previous state
        current_state = previous_state.clone()

        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.subplot(1, 2, 1)
        # plt.imshow(previous_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy().astype(np.uint8))
        # plt.subplot(1, 2, 2)
        # plt.imshow(current_state[:, -2, :, :, :].squeeze().permute(1, 2, 0).numpy().astype(np.uint8))
        # plt.show()

        # If the environment is a full MDP, get the whole world image
        if env.full_mdp:
            image = visual_field(env.world, color_map, channels=env.channels)
        # Otherwise, use the agent observation function
        else:
            image = visual_field(env.world, color_map, self.location, self.vision, channels=env.channels)

        # Update the latest state to the observation
        state_now = torch.tensor(image).unsqueeze(0)
        current_state[:, -1, :, :, :] = state_now

        return current_state
    
    def transition(self,
                   env: GridworldEnv) -> tuple:
        '''
        Changes the world based on the action taken.
        '''

        # Get current state
        state = self.pov(env)
        reward = 0

        # Take action based on current state
        action = self.model.take_action(state)

        # Attempt the transition 
        attempted_location = self.movement(action)
        target_object = env.observe(attempted_location)
        env.move(self, attempted_location)

        # Get the interaction reward
        reward += target_object.value 
        if hasattr(target_object, 'done'):
            done = True
        else:
            done = False

        # Get the next state   
        next_state = self.pov(env)

        # Increment the encounter list
        self.encounters[target_object.kind.lower()] += 1

        return state, action, reward, next_state, done
        
    def reset(self) -> None:
        self.episode_memory.clear()
        self.init_replay()
        # Reset encounters
        self.encounters = {
            'korean': 0,
            'lebanese': 0,
            'mexican': 0,
            'wall': 0,
            'emptyobject': 0
        }

    
# endregion
# ----------------------------------------------------- #