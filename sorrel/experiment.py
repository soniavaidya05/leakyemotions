from abc import abstractmethod
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from sorrel.agents import Agent
from sorrel.environments import GridworldEnv
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import ImageRenderer


class Experiment[E: GridworldEnv]:
    """An abstract wrapper class for running experiments with agents and environments.

    Attributes:
        env: The environment to run the experiment in.
        config: The configurations for the experiment.

            .. note::
                Some default methods provided by this class, such as `run`, require certain config parameters to be defined.
                These parameters are listed in the docstring of the method.
    """

    env: E
    config: DictConfig
    agents: list[Agent]

    def __init__(self, env: E, config: DictConfig | dict | list) -> None:

        if isinstance(config, DictConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            # note that if config is a list, we assume it is a dotlist
            self.config = OmegaConf.from_dotlist(config)

        self.setup_agents()

        self.env = env
        self.env.create_world()
        self.populate_environment()

    @abstractmethod
    def setup_agents(self) -> None:
        """This method should create a list of agents, and assign it to self.agents."""
        pass

    @abstractmethod
    def populate_environment(self) -> None:
        """This method should populate self.env.world.

        Note that self.env.world is already created with the specified dimensions, and
        every space is filled with the default entity of the environment, as part of
        self.env.create_world() when this experiment is constructed. One simply needs to
        place the agents and any additional entitites in self.env.world.
        """
        pass

    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents."""
        self.env.create_world()
        self.populate_environment()
        for agent in self.agents:
            agent.reset()

    # TODO: ability to save/load?
    def run(
        self, animate: bool = True, logging: bool = True, logger: Logger | None = None
    ) -> None:
        """Run the experiment.

        Required config parameters:
            - experiment.epochs: The number of epochs to run the experiment for.
            - experiment.max_turns: The maximum number of turns each epoch.
            - (Only if `animate` is true) experiment.record_period: The time interval at which to record the experiment.

        If `animate` is true,
        animates the experiment every `self.config.experiment.record_period` epochs.

        If `logging` is true, logs the total loss and total rewards each epoch.

        Args:
            animate: Whether to animate the experiment. Defaults to True.
            logging: Whether to log the experiment. Defaults to True.
            logger: The logger to use. Defaults to a ConsoleLogger.
        """
        renderer = None
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.env.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # start epoch action for each agent model
            for agent in self.agents:
                agent.model.start_epoch_action(epoch=epoch)

            # run the environment for the specified number of turns
            while not self.env.turn >= self.config.experiment.max_turns:
                # renderer should never be None if animate is true; this is just written for pyright to not complain
                if animate and renderer is not None:
                    renderer.add_image(self.env, epoch)
                self.env.take_turn()

            # generate the gif if animate is true
            if animate and renderer is not None:
                renderer.save_gif(epoch, Path(__file__).parent / "./data/")

            # At the end of each epoch, train the agents.
            total_loss = 0
            for agent in self.agents:
                loss = agent.model.train_step()
                total_loss += loss

            if logging:
                if not logger:
                    logger = ConsoleLogger(self.config.experiment.epochs)
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.env.total_reward,
                    self.agents[0].model.epsilon,
                )

            # update epsilon
            for agent in self.agents:
                agent.model.epsilon_decay(self.config.model.epsilon_decay)
