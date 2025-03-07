import numpy as np
from abc import abstractmethod
from typing import Sequence

from sorrel.buffers import ClaasyReplayBuffer

class SorrelModel:
  """
  Generic model class for Sorrel. All models should wrap around this implementation.
  """

  def __init__(
      self,
      input_size: int | Sequence[int],
      action_space: int,
      memory_size: int,
      epsilon: float = 0.
  ):
    
    self.input_size = input_size
    self.action_space = action_space
    _obs_for_input = input_size if isinstance(input_size, Sequence) else (input_size, )
    self.memory = ClaasyReplayBuffer(capacity=memory_size, obs_shape=_obs_for_input)
    self.epsilon = epsilon

  
  @abstractmethod
  def take_action(self, state: np.ndarray) -> int:
    """Take an action based on the observed input.
    Must be implemented by all subclasses of the model.
    
    Params:
      state: (np.ndarray) The observed input.

    Return:
      int: The action chosen.
    """
    pass

  def train_step(self) -> float | Sequence[float]:
    """Train the model.
    
    Return:
      float | Sequence[float]: The loss value.
    """
    return 0.
  
  def set_epsilon(self, new_epsilon: float) -> None:
    '''
    Replaces the current model epsilon with the provided value.
    '''
    self.epsilon = new_epsilon

  def start_epoch_action(self, **kwargs):
    """Actions to perform before each epoch."""
    pass

  def end_epoch_action(self, **kwargs):
    """Actions to perform after each epoch."""
    pass

  @property
  def model_name(self):
    """Get the name of the model class."""
    return self.__class__.__name__
