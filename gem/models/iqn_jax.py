# JAX Imports for automatic differentiation and numerical operations
from typing import Callable, Sequence
import jax
import jax.numpy as jnp
import jax.random
from flax import linen as nn

# Standard Python library imports for data structures and randomness
import heapq
from collections import deque
import random

# Numerical and array operations
import numpy as np

# Optimization library specific to JAX
import optax

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def compute_quantile_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    dones: jax.Array,
    critic: TrainState,
    gamma: float,
    key: jax.Array,
):
    key1, key2 = jax.random.split(key)
    value = critic.apply_fn(critic.params, state, key1)
    action_value = jnp.argmax(critic.apply_fn(critic.params, state, key2).mean(-1), -1, keepdims=True)
    value = jnp.take_along_axis(value, action_value, axis=-1)
    td_target = reward + gamma * value * (1 - dones)
    return td_target


def quantile_loss(pred, target, taus, mask):
    signed_loss = jnp.squeeze(target)[:, None, :] - jnp.squeeze(pred)[:, :, None]

    huber_loss = (
        # L2 loss when absolute error <= 1
        0.5 * jnp.int8((jnp.abs(signed_loss) <= 1)) * signed_loss**2
        +
        # L1 loss when absolute error > 1
        jnp.int8((jnp.abs(signed_loss) > 1)) * (jnp.abs(signed_loss) - 0.5)
    )

    quantile_errors = jnp.abs(taus - jnp.int8(signed_loss < 0))
    quantile_huber_loss = huber_loss * quantile_errors

    quantile_huber_loss = quantile_huber_loss * mask[:, None, None]

    return jnp.mean(quantile_huber_loss)


def quantile_bellman_residual_loss(
    state: jax.Array,
    quantile_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, Dict]:
    q, taus = critic.apply_fn(critic.params, state, key)

    critic_loss = quantile_loss(
        q, quantile_target, taus, mask
    )
    return (
        critic_loss,
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": q.mean(),
            "q1_quantiles_variance": q.squeeze().var(-1).mean(),
        },
    )



class ClaasyReplayBuffer:

    def __init__(self, capacity: int, obs_shape: Sequence[int]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.buffer = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
    
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

    def sample(self, batch_size, stacked_frames=1):
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
        indices = np.random.choice(self.size, batch_size - stacked_frames - 1, replace=False)
        indices = indices[:, np.newaxis]
        indices = (indices + np.arange(stacked_frames))

        states = self.buffer[indices]
        next_states = self.states[indices + 1]
        actions = self.actions[indices[:, -1]]
        rewards = self.rewards[indices[:, -1]]
        dones = self.dones[indices[:, -1]]
        # TODO [Claas]: revisit invalid computation, should be a bit wrong
        valid = 1. - np.any(self.dones[indices[:, :-1]], axis=-1)

        return states, actions, rewards, next_states, dones, valid


class IQNetwork(nn.Module):

    number_of_actions: int
    num_quantiles: int

    flatten_obs: bool = True

    activation_func: Callable[[jnp.array], jnp.array] = nn.relu
    obs_emb_layers: list = [256, 256]
    pi_emb_layers: list = [32, 32]
    shared_layers: list = [256]
    value_head: list = [256]
    advantage_head: list = [256]

    def cos_emb(self, bs, num_taus, key):
        # set up equal space frequencies
        pis = jnp.arange(0, 1, 1./num_taus)

        # sample random taus
        taus = jax.random.uniform(key, (bs, num_taus), minval=0, maxval=1)

        return jnp.cos(jnp.pi * taus * pis)


    @nn.compact
    def __call__(self, x, rng_key=None, noisy=False):
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        
        # compute obs embeddings
        for layer in self.obs_emb_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # Compute the cosine embeddings
        cos_emb = self.cos_emb(x.shape[0], self.num_quantiles, rng_key)
        for layer in self.pi_emb_layers:
            cos_emb = nn.Dense(layer)(cos_emb)
            cos_emb = self.activation_func(cos_emb)
        
        # combine multiplicatively
        x = x[:, None, :] * cos_emb[:, :, None]

        # shared embedding
        for layer in self.shared_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # compute value head
        for layer in self.value_head:
            value = nn.Dense(layer)(x)
            value = self.activation_func(value)
        value = nn.Dense(1)(value)

        # compute advantage head
        for layer in self.advantage_head:
            advantage = nn.Dense(layer)(x)
            advantage = self.activation_func(advantage)
        advantage = nn.Dense(self.number_of_actions)(advantage)

        # Combine value and advantage to get final Q-values
        # Subtracting the mean advantage (advantage.mean()) for stability
        q_values = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return q_values

class QNetwork(nn.Module):

    number_of_actions: int
    num_quantiles: int

    flatten_obs: bool = True

    activation_func: Callable[[jnp.array], jnp.array] = nn.relu
    emb_layers: list = [256, 256]
    value_head: list = [256]
    advantage_head: list = [256]

    @nn.compact
    def __call__(self, x, noisy=False):
        if self.flatten:
            x = x.reshape((x.shape[0], -1))
        
        # compute obs embeddings
        for layer in self.emb_layers:
            x = nn.Dense(layer)(x)
            x = self.activation_func(x)

        # compute value head
        for layer in self.value_head:
            value = nn.Dense(layer)(x)
            value = self.activation_func(value)
        value = nn.Dense(1)(value)

        # compute advantage head
        for layer in self.advantage_head:
            advantage = nn.Dense(layer)(x)
            advantage = self.activation_func(advantage)
        advantage = nn.Dense(self.number_of_actions)(advantage)

        # Combine value and advantage to get final Q-values
        # Subtracting the mean advantage (advantage.mean()) for stability
        q_values = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
        return q_values


class DoubleDQNAgent:
    """
    doubleDQN: An implementation of the Double Deep Q-Network (DDQN) algorithm.

    This class encapsulates the functionality of the Double DQN algorithm, an enhancement over the standard DQN.
    It addresses the overestimation of Q-values by decoupling the selection of actions from their evaluation.

    Attributes:
    - local_model: A QNetwork instance used for selecting actions and training.
    - target_model: A QNetwork instance used for evaluating the actions chosen by the local model.
    - replay_buffer: A buffer for storing experiences. Can be either standard or prioritized (PER).
    - per: A boolean indicating whether to use Prioritized Experience Replay (PER).
    - optimizer: An optimizer for updating the local model's parameters.
    - local_model_params: The parameters of the local model.
    - target_model_params: The parameters of the target model.
    - optimizer_state: The state of the optimizer.

    Initialization:
    - The constructor initializes two QNetwork instances: one as the local model and the other as the target model.
    - It also initializes a replay buffer, which can be either a standard buffer or a prioritized buffer based on the `per` flag.
    - The models are initialized with dummy data to set up their parameters. The shape of the dummy data should match the expected input shape.
    - An optimizer (Adam in this case) is initialized with the parameters of the local model.

    Usage:
    - The local model is used to select actions and is regularly trained with experiences from the replay buffer.
    - The target model's parameters are periodically updated with those of the local model,
      reducing the overestimation bias inherent in standard DQN.
    - If `per` is set to True, the replay buffer prioritizes experiences based on their TD error,
      potentially leading to more efficient learning.

    Notes:
    - The separation of action selection (local model) and action evaluation (target model) is key to the Double DQN approach.
    - The use of PER is optional and can be beneficial in complex environments where some experiences are significantly more important.
    - The models and optimizer should be appropriately configured to match the specifics of the environment and task.
    - Regular synchronization of the local and target models is crucial for the algorithm's stability and performance.

    Example:
    ```
    model = doubleDQN(per=True)
    # Use model.local_model and agent.target_model for training and action selection
    ```
    """

    def __init__(
        self,
        input_size=1134,
        number_of_actions=4,
        lr=0.001,
        gamma=0.99,
        per=False,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0006,
        capacity=5000,
        seed=0
    ):
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        self.lr = lr
        self.per = per
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = capacity

        self.local_model = QNetwork(number_of_actions=number_of_actions)
        self.target_model = QNetwork(number_of_actions=number_of_actions)

        # Initialize RNG key for the model
        self.rng_key = jax.random.PRNGKey(seed)

        self.replay_buffer = ClaasyReplayBuffer(capacity=capacity)

        # Initialize models with dummy data
        dummy_input = jnp.zeros((1, input_size))  # Should be part of the constructor

        # Initialize the parameters of the models
        self.local_model_params = self.local_model.init(
            jax.random.PRNGKey(seed), dummy_input
        )["params"]
        self.target_model_params = self.target_model.init(
            jax.random.PRNGKey(seed+1), dummy_input
        )["params"]

        # Initialize the optimizer with the parameters of the local model
        self.optimizer = optax.adam(lr)

        self.train_state: TrainState = TrainState.create(
            apply_fn=self.local_model.apply,
            params=self.local_model_params,
            tx=self.optimizer,
        )
        self.target_train_state: TrainState = TrainState.create(
            apply_fn=self.target_model.apply,
            params=self.target_model_params,
            tx=optax.set_to_zero,
        )

    def take_action(self, state, epsilon):
        """
        Selects an action based on the current state, using an epsilon-greedy strategy.

        This method decides between exploration and exploitation using the epsilon value.
        With probability epsilon, it chooses a random action (exploration),
        and with probability 1 - epsilon, it chooses the best action based on the model's predictions (exploitation).

        Parameters:
        - state: The current state of the environment.
        - epsilon: The probability of choosing a random action.
        - rng: The JAX random number generator key.

        Things to Check:
        - Ensure that the RNG key is split or advanced correctly instead of being reseeded each time.
        In JAX, you typically use `jax.random.split` to get a new subkey.
        - Validate that the state is correctly reshaped and can be processed by the model during exploitation.
        - Confirm that the action space defined by `number_of_actions` aligns with the environment's action space.
        - Check that the `jax.random.uniform` and `jax.random.randint` functions are used correctly to 
          generate random numbers and actions.

        Potential Improvements:
        - Modify the RNG handling to use `jax.random.split` for generating new keys.
        This will maintain the stochasticity without the need to reset the seed each time.
        - If the state requires specific preprocessing before being fed into the model,
        ensure that it's handled appropriately within this method or prior to calling it.
        - Consider parameterizing the number of actions if it varies across different environments or parts of your application.

        Returns:
        - The selected action, either random or the best according to the model.
        """

        # Split the RNG key
        self.rng_key, rng_key_action, rng_key_taus = jax.random.split(self.rng_key, 3)

        # Generate a random number using JAX's random number generator
        random_number = jax.random.uniform(rng_key_action, shape=())

        if random_number < epsilon:
            # Exploration: choose a random action
            self.rng_key, rng_key_random = jax.random.split(self.rng_key)
            action = jax.random.randint(
                rng_key_random, shape=(), minval=0, maxval=self.number_of_actions
            )
        else:
            # Exploitation: choose the best action based on model prediction
            q_values = self.train_state.apply_fn(self.train_state.params, state, rng_key_taus)
            action = jnp.argmax(jnp.mean(q_values, axis=-1), axis=-1)[0]

        return action

    def compute_td_errors(self, params, states, actions, rewards, next_states, dones):
        q_values = self.train_state.apply_fn(params, states)
        next_q_values = self.target_model.apply({"params": params}, next_states)
        max_next_q_values = jnp.max(next_q_values, axis=1)
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        actions_one_hot = jax.nn.one_hot(actions, self.number_of_actions)
        predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
        return jnp.abs(predicted_q_values - target_q_values)

    def train_step(self, batch_size, soft_update=True):
        """
        Perform a training step, with control over batch size, discount factor, and update type of the target model.

        Parameters:
        - batch_size: Determines the number of samples to be used in each training step. A larger batch size
        generally leads to more stable gradient estimates, but it requires more memory and computational power.
        - gamma: The discount factor used in the calculation of the return. It determines the importance of
        future rewards. A value of 0 makes the agent short-sighted by only considering current rewards, while
        a value close to 1 makes it strive for long-term high rewards.
        - soft_update: A boolean flag that controls the type of update performed on the target model's parameters.
        If True, a soft update is performed, where the target model parameters are gradually blended with the
        local model parameters. If False, a hard update is performed, directly copying the local model parameters
        to the target model.

        The function conducts a single step of training, which involves sampling a batch of experiences,
        computing the loss, updating the model parameters based on the computed gradients, and then updating
        the target model parameters.

        The choice between soft and hard updates for the target model is crucial for the stability of the training process.
        Soft updates provide more stable but slower convergence, while hard updates can lead to faster convergence
        but might cause instability in training dynamics.
        """

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        dones = dones.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        def loss_fn(params):
            q_values = self.local_model.apply({"params": params}, states)
            next_q_values = self.target_model.apply({"params": params}, next_states)
            max_next_q_values = jnp.max(next_q_values, axis=1)
            target_q_values = rewards + (
                self.gamma * max_next_q_values * (1 - dones)
            )
            actions_one_hot = jax.nn.one_hot(actions, self.number_of_actions)
            predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
            loss = jnp.mean((predicted_q_values - target_q_values) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, gradients = grad_fn(self.local_model_params)
        updates, self.optimizer_state = self.optimizer.update(
            gradients, self.optimizer_state
        )
        self.local_model_params = optax.apply_updates(
            self.local_model_params, updates
        )

        # Soft update

        tau = 0.01
        self.target_model_params = jax.tree_map(
            lambda t, l: tau * l + (1 - tau) * t,
            self.target_model_params,
            self.local_model_params,
        )

        return loss

    def hard_update(self):
        """
        Perform a hard update on the target model parameters.

        This function directly copies the parameters from the local model to the
        target model. Unlike a soft update, which gradually blends the parameters
        of the local and target models, a hard update sets the target model
        parameters to be exactly equal to the local model parameters. This is
        typically done less frequently than soft updates and can be used to
        periodically reset the target model with the latest local model parameters.

        Usage:
            This function should be called at specific intervals (e.g., every 100
            epochs) during the training process, depending on the specific needs
            and dynamics of the training regimen.

        Note:
            Frequent hard updates can lead to more volatile training dynamics,
            whereas infrequent updates might cause the target model to lag too far
            behind the local model. The frequency of hard updates should be chosen
            carefully based on empirical results and the specific characteristics
            of the model and training data.
        """
        self.target_model_params = jax.tree_map(
            lambda l: l,
            self.local_model_params,
        )


class IQNAgent:
    """
    doubleDQN: An implementation of the Double Deep Q-Network (DDQN) algorithm.

    This class encapsulates the functionality of the Double DQN algorithm, an enhancement over the standard DQN.
    It addresses the overestimation of Q-values by decoupling the selection of actions from their evaluation.

    Attributes:
    - local_model: A QNetwork instance used for selecting actions and training.
    - target_model: A QNetwork instance used for evaluating the actions chosen by the local model.
    - replay_buffer: A buffer for storing experiences. Can be either standard or prioritized (PER).
    - per: A boolean indicating whether to use Prioritized Experience Replay (PER).
    - optimizer: An optimizer for updating the local model's parameters.
    - local_model_params: The parameters of the local model.
    - target_model_params: The parameters of the target model.
    - optimizer_state: The state of the optimizer.

    Initialization:
    - The constructor initializes two QNetwork instances: one as the local model and the other as the target model.
    - It also initializes a replay buffer, which can be either a standard buffer or a prioritized buffer based on the `per` flag.
    - The models are initialized with dummy data to set up their parameters. The shape of the dummy data should match the expected input shape.
    - An optimizer (Adam in this case) is initialized with the parameters of the local model.

    Usage:
    - The local model is used to select actions and is regularly trained with experiences from the replay buffer.
    - The target model's parameters are periodically updated with those of the local model,
      reducing the overestimation bias inherent in standard DQN.
    - If `per` is set to True, the replay buffer prioritizes experiences based on their TD error,
      potentially leading to more efficient learning.

    Notes:
    - The separation of action selection (local model) and action evaluation (target model) is key to the Double DQN approach.
    - The use of PER is optional and can be beneficial in complex environments where some experiences are significantly more important.
    - The models and optimizer should be appropriately configured to match the specifics of the environment and task.
    - Regular synchronization of the local and target models is crucial for the algorithm's stability and performance.

    Example:
    ```
    model = doubleDQN(per=True)
    # Use model.local_model and agent.target_model for training and action selection
    ```
    """

    def __init__(
        self,
        input_size=1134,
        number_of_actions=4,
        lr=0.001,
        gamma=0.99,
        per=False,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0006,
        capacity=5000,
        seed=0
    ):
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        self.lr = lr
        self.per = per
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = capacity

        self.local_model = IQNetwork(number_of_actions=number_of_actions)
        self.target_model = IQNetwork(number_of_actions=number_of_actions)

        # Initialize RNG key for the model
        self.rng_key = jax.random.PRNGKey(seed)

        self.replay_buffer = ClaasyReplayBuffer(capacity=capacity)

        # Initialize models with dummy data
        dummy_input = jnp.zeros((1, input_size))  # Should be part of the constructor

        # Initialize the parameters of the models
        self.local_model_params = self.local_model.init(
            jax.random.PRNGKey(seed), dummy_input, self.rng_key
        )["params"]
        self.target_model_params = self.target_model.init(
            jax.random.PRNGKey(seed+1), dummy_input, self.rng_key
        )["params"]

        # Initialize the optimizer with the parameters of the local model
        self.optimizer = optax.adam(lr)

        self.train_state: TrainState = TrainState.create(
            apply_fn=self.local_model.apply,
            params=self.local_model_params,
            tx=self.optimizer,
        )
        self.target_train_state: TrainState = TrainState.create(
            apply_fn=self.target_model.apply,
            params=self.target_model_params,
            tx=optax.set_to_zero,
        )

    def take_action(self, state, epsilon):
        """
        Selects an action based on the current state, using an epsilon-greedy strategy.

        This method decides between exploration and exploitation using the epsilon value.
        With probability epsilon, it chooses a random action (exploration),
        and with probability 1 - epsilon, it chooses the best action based on the model's predictions (exploitation).

        Parameters:
        - state: The current state of the environment.
        - epsilon: The probability of choosing a random action.
        - rng: The JAX random number generator key.

        Things to Check:
        - Ensure that the RNG key is split or advanced correctly instead of being reseeded each time.
        In JAX, you typically use `jax.random.split` to get a new subkey.
        - Validate that the state is correctly reshaped and can be processed by the model during exploitation.
        - Confirm that the action space defined by `number_of_actions` aligns with the environment's action space.
        - Check that the `jax.random.uniform` and `jax.random.randint` functions are used correctly to 
          generate random numbers and actions.

        Potential Improvements:
        - Modify the RNG handling to use `jax.random.split` for generating new keys.
        This will maintain the stochasticity without the need to reset the seed each time.
        - If the state requires specific preprocessing before being fed into the model,
        ensure that it's handled appropriately within this method or prior to calling it.
        - Consider parameterizing the number of actions if it varies across different environments or parts of your application.

        Returns:
        - The selected action, either random or the best according to the model.
        """

        def action_fn(state, model: TrainState, key, number_of_actions, epsilon):
            # Split the RNG key
            rng_key, rng_key_action, rng_key_taus = jax.random.split(key, 3)

            # Generate a random number using JAX's random number generator
            random_number = jax.random.uniform(rng_key_action, shape=())

            if random_number < epsilon:
                # Exploration: choose a random action
                rng_key, rng_key_random = jax.random.split(rng_key)
                action = jax.random.randint(
                    rng_key_random, shape=(), minval=0, maxval=number_of_actions
                )
            else:
                # Exploitation: choose the best action based on model prediction
                q_values = model.apply_fn(model.params, state, rng_key_taus)
                action = jnp.argmax(jnp.mean(q_values, axis=-1), axis=-1)[0]

            return action, rng_key
        
        action, self.rng_key = jax.jit(action_fn, static_argnums=(3))(
            state, self.train_state, self.rng_key, self.number_of_actions, epsilon)
        return action


    def train_step(self, batch_size, soft_update=True):
        """
        Perform a training step, with control over batch size, discount factor, and update type of the target model.

        Parameters:
        - batch_size: Determines the number of samples to be used in each training step. A larger batch size
        generally leads to more stable gradient estimates, but it requires more memory and computational power.
        - gamma: The discount factor used in the calculation of the return. It determines the importance of
        future rewards. A value of 0 makes the agent short-sighted by only considering current rewards, while
        a value close to 1 makes it strive for long-term high rewards.
        - soft_update: A boolean flag that controls the type of update performed on the target model's parameters.
        If True, a soft update is performed, where the target model parameters are gradually blended with the
        local model parameters. If False, a hard update is performed, directly copying the local model parameters
        to the target model.

        The function conducts a single step of training, which involves sampling a batch of experiences,
        computing the loss, updating the model parameters based on the computed gradients, and then updating
        the target model parameters.

        The choice between soft and hard updates for the target model is crucial for the stability of the training process.
        Soft updates provide more stable but slower convergence, while hard updates can lead to faster convergence
        but might cause instability in training dynamics.
        """

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, valid = batch

        def update_fn(states, actions, rewards, next_states, dones, valid, model: TrainState, target_model: TrainState, keys):
            next_key, key = jax.random.split(keys)
            dones = dones.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)

            def loss_fn(params):
                quantile_target = compute_quantile_td_target_from_state(
                    next_states, rewards, dones, target_model, self.gamma, self.rng_key
                )
                loss, loss_dict = quantile_bellman_residual_loss(
                    states, quantile_target, model, valid, self.rng_key

                )

                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, gradients = grad_fn(model.params)
            model = model.apply_gradients(grads=gradients)

            # Soft update

            tau = 0.01
            target_model.params = jax.tree_map(
                lambda t, l: tau * l + (1 - tau) * t,
                target_model.params,
                model.params,
            )

            return loss, next_key, model, target_model
        
        loss, self.rng_key, self.train_state, self.target_train_state = jax.jit(jax.vmap(update_fn, in_axes=(
            0, 0, 0, 0, 0, 0, None, None, None)))(
            states, actions, rewards, next_states, dones, valid, self.train_state, self.target_train_state, self.rng_key)
        return loss

    def hard_update(self):
        """
        Perform a hard update on the target model parameters.

        This function directly copies the parameters from the local model to the
        target model. Unlike a soft update, which gradually blends the parameters
        of the local and target models, a hard update sets the target model
        parameters to be exactly equal to the local model parameters. This is
        typically done less frequently than soft updates and can be used to
        periodically reset the target model with the latest local model parameters.

        Usage:
            This function should be called at specific intervals (e.g., every 100
            epochs) during the training process, depending on the specific needs
            and dynamics of the training regimen.

        Note:
            Frequent hard updates can lead to more volatile training dynamics,
            whereas infrequent updates might cause the target model to lag too far
            behind the local model. The frequency of hard updates should be chosen
            carefully based on empirical results and the specific characteristics
            of the model and training data.
        """
        self.target_model_params = jax.tree_map(
            lambda l: l,
            self.local_model_params,
        )

