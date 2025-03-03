# How to implement a model in Agentarium

This tutorial provides an introduction to wrapping a model for use with Agentarium. While this tutorial will provide an introduction to the base model interface, Agentarium also includes existing pre-built IQN and DDQN models, interfaces for models written in Jax or PyTorch, and a simple interface for human input.

For more information about the base model classes, please consult the documentation for the `AgentariumModel` class [../reference/models.html](here).

## Base model functions

The `AgentariumModel` is an abstract class intended to be as minimal and flexible as possible. It can wrap any kind of model as long as it implements a method for taking an action, by overriding the method `take_action()`:

```python
  @abstractmethod
  def take_action(self, state: np.ndarray) -> int:
    """Take an action based on the observed input.
    Must be implemented by all subclasses of the model.
    
    Params:
      state: (np.ndarray) The observed input.

    Return:
      int: The action chosen.
    """
```

At a minimum, this method is called whenever an agent takes an action in the environment using the `Agent.transition()` function. It can additionally be called when a model completes a training step using the `train_step()` function (described below). The base class takes a single input, a state (generated from an environment when an agent makes an observation using `agentarium.observation.observation_spec.ObservationSpec.observe()`) and yields an output `int` that corresponds to the action taken.

Many models have an exploration parameter :math:`\varepsilon` that dictate the chance of taking a random action. The base model has a convenience function `set_epsilon()` to allow this value to be updated across time:

```python
  def set_epsilon(self, new_epsilon: float) -> None:
    '''
    Replaces the current model epsilon with the provided value.
    '''
    self.epsilon = new_epsilon
```

### Trainable models

Although not all models require training (e.g., a model that implements a preselected action policy, or a model that accepts human input), it also supports wrapping the model training loop using the function `train_step()`:

```python
  def train_step(self) -> float | Sequence[float] | torch.Tensor | jax.Array:
    """Train the model.
    
    Return:
      float | Sequence[float]: The loss value.
    """
    return 0.
```

Custom implementations can return a single loss value or a sequence of values, including Jax arrays and PyTorch tensors. By default, the function returns 0 if no specific loss procedure is computed.

### Additional functions

Some models require additional updates or changes outside of model training (e.g., to update parameters when using both a local and target network). The base class implements two functions `start_epoch_action()` and `end_epoch_action()` that can perform any similar actions before or after an epoch.

```python
  def start_epoch_action(self, **kwargs):
    """Actions to perform before each epoch."""
    pass

  def end_epoch_action(self, **kwargs):
    """Actions to perform after each epoch."""
    pass
```

### Using the replay buffer

By default, `AgentariumModel` has a replay buffer `Agentarium.buffers.ClaasyReplayBuffer` that can store up to `memory_size` states. This buffer can allow the model to:
- Add states to the model's replay buffer
- Sample minibatches from the model (e.g., for model training)
- Get the current state (including 'stacked' frames if the model's input includes more than one frame at a time)

Experiences in the model are stored using a :math:`(S, A, R, D)` format using the function `self.memory.add()`:

```python
    def add(self, obs, action, reward, done):
        """
        Add an experience to the replay buffer.

        Args:
            obs (np.ndarray): The observation/state.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode terminated after this step.
        """
        self.buffer[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
```

To sample the replay buffer, the function `self.memory.sample()` is used:

```python
    def sample(self, batch_size: int, stacked_frames: int = 1):
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.
            stacked_frames (int): The number of frames to stack together.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                A tuple containing the states, actions, rewards, next states, dones, and 
                invalid (meaning stacked frmaes cross episode boundary).
        """
        indices = np.random.choice(max(1, self.size - stacked_frames - 1),  batch_size, replace=False)
        indices = indices[:, np.newaxis]
        indices = (indices + np.arange(stacked_frames))

        states = torch.from_numpy(self.buffer[indices]).view(batch_size, -1)
        next_states = torch.from_numpy(self.buffer[indices + 1]).view(batch_size, -1)
        actions = torch.from_numpy(self.actions[indices[:, -1]]).view(batch_size, -1)
        rewards  = torch.from_numpy(self.rewards[indices[:, -1]]).view(batch_size, -1)
        dones = torch.from_numpy(self.dones[indices[:, -1]]).view(batch_size, -1)
        valid = torch.from_numpy(1. - np.any(self.dones[indices[:, :-1]], axis=-1)).view(batch_size, -1)

        return states, actions, rewards, next_states, dones, valid
```

This function returns a batch of size `batch_size` in the format :math:`(S, A, R, S', D)`.

```{eval-rst}
.. note::
   Because samples may sometimes be drawn on the boundaries of episodes, an additional parameter :code:`valid` is returned here. This ensured that if losses are computed from this sample, "invalid" batches (i.e., those crossing episode boundaries) do not affect the losses.
```