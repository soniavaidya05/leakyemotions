
import abc

import torch

from environment import GridworldEnv
from entity import Entity
from gem.config import Cfg


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

        # TODO: episode_memory/Memory class required for every agent? trucks=RPG implementations fine?
        # Memory will be property of the model instead of the Agent class
        # the Agent should have an add_memory function (ref cleanup agent)
        # want an abstract Model class (for functions like memory.clear / reset)
        # TODO: self.num_memories? = cfg.agent.agent.memory_size (as in cleanup) or .num_memories (as in RPG)?

    # TODO: memory (LSTM) and therefore replay needs to be initialized for every agent?
    # answer: put this in the Memory class
    # @abc.abstractmethod
    # def init_replay(self) -> None:
    #     pass

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
