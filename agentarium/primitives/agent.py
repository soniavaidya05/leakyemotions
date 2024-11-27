import abc
from typing import Any

import numpy as np
import torch

from agentarium.config import Cfg
from agentarium.primitives.environment import Entity, GridworldEnv


class Agent(Entity):
    """
    An abstract class for agents, a special type of entities. Note that this is a subclass of Entity.

    Attributes:
        - :attr:`cfg` - The configuration to use for this agent.
        - :attr:`model` - The model that this agent uses.
        - :attr:`action_space` - The range of actions that the agent is able to take, represented by a list of integers.

            .. warning::
                Currently, each element in :code:`action_space` should be the index of that element.
                In other words, it should be a list of neighbouring integers in increasing order starting at 0.

                For example, if the agent has 4 possible actions, it should have :code:`action_space = [0, 1, 2, 3]`.

    Attributes that override parent (Entity)'s default values:
        - :attr:`vision` - set at time of initialization based on the cfg provided, instead of defaulting to None.
        - :attr:`has_transitions` - Defaults to True instead of False.
    """

    cfg: Cfg
    model: Any
    action_space: list[int]

    def __init__(self, cfg: Cfg, appearance, model, action_space, location=None):
        # initializations based on parameters
        self.cfg = cfg
        self.model = model
        self.action_space = action_space
        self.location = location

        super().__init__(appearance)

        # overriding parent default attributes
        self.vision = cfg.agent.agent.vision
        self.has_transitions = True

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        """Add an experience to the memory.

        Args:
            state (np.ndarray): the state to be added.
            action (int): the action taken by the agent.
            reward (float): the reward received by the agent.
            done (bool): whether the episode terminated after this experience.
        """
        self.model.memory.add(state, action, reward, done)

    @abc.abstractmethod
    def act(self, action) -> tuple[int, ...]:
        """Act on the environment.

        Args:
            action: an element from this agent's action space indicating the action to take.

        Returns:
            tuple[int, ...]: A location tuple indicating the updated location of the agent.
        """
        pass

    @abc.abstractmethod
    def pov(self, env: GridworldEnv) -> torch.Tensor:
        """
        Defines the agent's observation function.

        Args:
            env (GridworldEnv): the environment that this agent is observing.

        Returns:
            torch.Tensor: the observed information.
        """
        pass

    @abc.abstractmethod
    def transition(self, env: GridworldEnv, state, action) -> torch.Tensor:
        """
        Changes the environment based on action taken by the agent.

        Args:
            env (GridworldEnv): the environment that this agent is acting in.
            state: the current state.
            action: an element from this agent's action space indicating the action taken.

        Returns:
            torch.Tensor: the result of the transition.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the agent (and its memory).
        """
        pass
