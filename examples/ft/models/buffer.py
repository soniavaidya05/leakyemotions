# --------------- #
# region: Imports #
# --------------- #

from __future__ import annotations
# Import base packages
import torch
import numpy as np
import random
import dill
import os
from collections import deque, namedtuple
from typing import Union

# --------------- #
# endregion       #
# --------------- #

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.

        Parameters:
            buffer_size (int): maximum size of buffer \n
            batch_size (int): size of each training batch \n
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.gamma = gamma
        # self.n_step = n_step
        # self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        # # Add the new experience to buffer
        # self.n_step_buffer.append((state, action, reward, next_state, done))

        # # If there are enough steps in the buffer, append to the memory
        # if len(self.n_step_buffer) == self.n_step:

        #     # Set the experience as the return and state change over multiple steps 
        #     state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
        #     e = self.experience(state, action, reward, next_state, done)

        #     # Add the experience to the memory
        #     self.memory.append(e)

        # NOTE: multistep return seem to have little/negative effect on the performance
        # NOTE: removing multistep return also bounds the loss to a lower number
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # region: hidden func
    # def calc_multistep_return(self, n_step_buffer):
    #     Return = 0
    #     for idx in range(self.n_step):
    #         Return += self.gamma**idx * n_step_buffer[idx][2]

    #     # There are 3 steps in the buffer
    #     # - state = state of first step
    #     # - action = action of first step
    #     # - reward = sum of rewards of all steps
    #     # - next_state = state of last step
    #     # - done = done of last step
    
    #     return (
    #         n_step_buffer[0][0],
    #         n_step_buffer[0][1],
    #         Return,
    #         n_step_buffer[-1][3],
    #         n_step_buffer[-1][4],
    #     )
    # endregion

    def sample(self) -> tuple[torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size if self.batch_size < len(self) else len(self.memory))

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(current_size={len(self)},maximum_size={self.memory.maxlen},batch_size={self.batch_size})'
    
    def save(self, file_path: Union[str, os.PathLike]):
        '''
        Saves the current experience buffer along with its memories to disk.

        Parameters:
            file_path: The path to save the memories to.
        '''
        with open(file_path, 'wb') as path:
            dill.dump(self, path, protocol=dill.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, file_path: Union[str, os.PathLike]) -> ReplayBuffer:
        '''
        Loads an experience buffer along with its memories from disk.

        Parameters:
            file_path: The path to load the memories from.
        '''
        with open(file_path, 'rb') as path:
            memories = dill.load(path)
            return memories


class ActionBatchReplayBuffer(ReplayBuffer):

    '''
    Container for batching actions into the appropriate number of timesteps for the BC model.

    Parameters:
        action_space: The number of possible actions PLUS ONE. \n
        timesteps: The number of timesteps to store. \n
    '''

    def __init__(
        self,
        buffer_size,
        batch_size,
        device,
        seed,
        gamma,
        timesteps,
        action_space
    ):
        super(ActionBatchReplayBuffer, self).__init__(buffer_size, batch_size, device, seed, gamma)
        self.timesteps = timesteps
        self.action_space = action_space
        self.action_batch = deque([],maxlen=timesteps)
        self.reset_action_batch()

    def reshape(
        self,
        action: int
    ) -> np.ndarray:
        '''
        Reshapes the action input into a one-hot coded vector.
        '''
        return np.array([1 if i == action else 0 for i in range(self.action_space)])
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        # Reshape the action to a one-hot code
        action = self.reshape(action)
        # Append to the deque
        self.action_batch.append(action)
        # Convert to numpy array
        actions = torch.from_numpy(np.array(self.action_batch)).unsqueeze(0)

        # Add the batched action along with the rest of the experience tuple
        e = self.experience(state, actions, reward, next_state, done)
        self.memory.append(e)

    def reset_action_batch(self):
        '''
        Empty out the action deque so that the initial actions for the next
        game are dummy actions.
        '''
        for _ in range(self.timesteps):
            # Fill the action space with dummy actions
            self.action_batch.append(self.reshape(self.action_space))

    def get_game(self):
        return self.experience