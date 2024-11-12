
import abc

import torch
import numpy as np

from agentarium.primitives.environment import GridworldEnv, Entity
from agentarium.config import Cfg


class Agent(Entity):
    """Abstract agent class."""
    def __init__(self, cfg: Cfg, appearance, model, action_space, location = None):

        # initializations based on parameters
        self.cfg = cfg
        self.model = model
        self.action_space = action_space
        self.location = location

        super.__init__(appearance)

        # overriding parent default attributes
        self.vision = cfg.agent.agent.vision
        self.has_transitions = True

        # TODO: Memory will be property of the model instead of the Agent class
        # -> does every model need a memory?

    @abc.abstractmethod
    def act(self, action) -> tuple[int, ...]:
        """Act on the environment.

        Params:
            action: an element from this agent's action space indicating the action to take.

        Return:
            tuple[int, ...] A location tuple indicating the updated location of the agent.
        """
        pass

    @abc.abstractmethod
    def pov(self, env: GridworldEnv) -> torch.Tensor:
        """
        Defines the agent's observation function.
        """
        pass

    @abc.abstractmethod
    def transition(self, env: GridworldEnv, state, action) -> torch.Tensor:
        """
        Changes the environment based on action taken by the agent.
        """
        pass

    # TODO: leave as implemented or change to abstract?
    def add_memory(self, state: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an experience to the memory."""
        self.model.memory.add(state, action, reward, done)

    @abc.abstractmethod
    def reset(self) -> None:
        pass
