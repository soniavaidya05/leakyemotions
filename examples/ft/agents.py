from examples.ft.entities import Object, EmptyObject
from collections import deque
import torch

# ----------------------------------------------------- #
# region: Memory class                                  #

class Memory:
    '''
    Memory for the agent class
    '''

    def __init__(self, memory_size: int):
        self.priorities = deque(maxlen=memory_size)
        self.states = deque(maxlen=memory_size)
        self.actions = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.nextstates = deque(maxlen=memory_size)
        self.dones = deque(maxlen=memory_size)

    def clear(self):
        del self.priorities[:]
        del self.actions[:]
        del self.states[:]
        del self.nextstates[:]
        del self.rewards[:]
        del self.dones[:]

    def append(self, exp):
        '''
        Add an experience to the agent's memory.
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

    def get_last_memory(self, attr: str = None):
        '''
        Get the latest memory from the replay.
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

                        Parameters
                        ----------
    color: The appearance of the agent (list of N floats)
    model: The model object
    memory_size: The size of the replay buffer
    '''
    def __init__(self, 
                 color: list, 
                 model, 
                 vision: int = 4, 
                 memory_size: int = 100
                ):
        super().__init__(color)
        self.vision = vision
        self.model = model
        self.location = None
        self.has_transitions = True
        self.episode_memory = Memory(memory_size)

    def init_replay(self, 
                    num_frames: int = 5, 
                    state_shape = None
                    ):
        '''
        Fill in blank images for the LSTM. Requires the state size to be fully defined.

                                        Parameters
                                        ----------
        num_frames: The number of frames per state observation
        state_shape: (Optional) a tuple or list of C x H x W of the state size.
        If it is not specified, the state size will be specified with the agent's vision.
        '''
        priority = torch.tensor(0.1)

        if state_shape is not None:
            state = torch.zeros(1, num_frames, *state_shape).float()
        else:
            C = len(self.color)
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
                 ):
        
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
            env):
        '''
        Defines the agent's observation function
        '''
        # Get the previous state
        previous_state = self.episode_memory.get_last_memory('states')

        # Get the last four frames from the previous state
        current_state = previous_state.clone()
        current_state[:, 0:-1, :, :, :] = previous_state[:, 1:, :, :, :]

        


    
    def transition(self,
                   action: int,
                   env: Env):
        '''
        Changes the world based on the action taken.
        '''
        reward = 0
        attempted_location = self.movement(action)
        interaction = env.world[attempted_location]

        # Get the interaction reward
        reward += interaction.value 

        if interaction.passable:
            # Move the agent to the new location
            env.world[attempted_location] = self 
            self.location = attempted_location
            # Replace the agent from the old location
            env.world[self.location] = env.default_object 

        
    
# endregion
# ----------------------------------------------------- #