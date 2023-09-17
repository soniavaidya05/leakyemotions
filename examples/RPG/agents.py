from examples.RPG.entities import Wall, EmptyObject
from examples.trucks.agents import Memory

from ast import literal_eval as make_tuple
from typing import Optional
from numpy.typing import ArrayLike
import torch

from gem.utils import visual_field



# TODO: 
# make sure dead agent should change model to None
# probably need to add die into particular circumstances in transition
# touch base with Eric about adding in new pov code

class Agent:
    def __init__(self, model, cfg):
        self.kind = "Agent"
        self.cfg = cfg      
        self.appearance = make_tuple(cfg.agent.agent.appearance)  # agents are blue
        self.tile_size = make_tuple(cfg.agent.agent.tile_size)
        self.sprite = f'{cfg.root}/examples/RPG/assets/hero.png'
        self.passable = 0  # whether the object blocks movement
        self.value = 0  # agents have no value
        self.health = cfg.agent.agent.health  # for the agents, this is how hungry they are
        self.location = None
        self.action_type = "neural_network"
        self.vision = cfg.agent.agent.vision
        
        # training-related features
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.episode_memory = Memory(cfg.agent.agent.memory_size)
        self.num_memories = cfg.agent.agent.num_memories
        self.init_rnn_state = None
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0
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
        if action == 0: # UP
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-back.png'
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1: # DOWN
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero.png'
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2: # LEFT
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-left.png'
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3: # RIGHT
            self.sprite = f'{self.cfg.root}/examples/RPG/assets/hero-right.png'
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        return new_location
    
    def pov(self,
            env) -> torch.Tensor:
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
            image = visual_field(env.world, color_map, channels=self.cfg.model.iqn.parameters.state_size[0])
        # Otherwise, use the agent observation function
        else:
            image = visual_field(env.world, color_map, self.location, self.vision, channels=self.cfg.model.iqn.parameters.state_size[0])

        # Update the latest state to the observation
        state_now = torch.tensor(image).unsqueeze(0)
        current_state[:, -1, :, :, :] = state_now

        return current_state
    
    def transition(self,
                   env) -> tuple:
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

        # Add to the encounter record
        if str(target_object) in self.encounters.keys():
            self.encounters[str(target_object)] += 1 

        # Get the next state   
        next_state = self.pov(env)

        return state, action, reward, next_state, False
        
    def reset(self) -> None:
        self.episode_memory.clear()
        self.init_replay()
        self.encounters = {
            'Gem': 0,
            'Coin': 0,
            'Food': 0,
            'Bone': 0,
            'Wall': 0
        }

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

def color_map(channels: int) -> dict:
    '''
    Generates a color map for the food truck environment.

    Parameters:
        channels: the number of appearance channels in the environment

    Return:
        A dict of object-color mappings
    '''
    if channels > 5:
        colors = {
            'EmptyObject': [0 for _ in range(channels)],
            'Agent': [255 if x == 0 else 0 for x in range(channels)],
            'Wall': [255 if x == 1 else 0 for x in range(channels)],
            'Gem': [255 if x == 2 else 0 for x in range(channels)],
            'Food': [255 if x == 3 else 0 for x in range(channels)],
            'Coin': [255 if x == 4 else 0 for x in range(channels)],
            'Bone': [255 if x == 5 else 0 for x in range(channels)]
        }
    else:
        colors = {
            'EmptyObject': [255., 255., 255.],
            'Agent': [0., 0., 255.],
            'Wall': [153.0, 51.0, 102.0],
            'Gem': [0., 255., 0.],
            'Coin': [255., 255., 0.],
            'Food': [255., 0., 0.],
            'Bone': [0., 0., 0.]
        }
    return colors