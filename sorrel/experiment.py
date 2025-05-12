from abc import abstractmethod
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from sorrel.agents import Agent
from sorrel.environments import GridworldEnv
from sorrel.utils.logging import ConsoleLogger, Logger
from sorrel.utils.visualization import animate, image_from_array, render_sprite


class Experiment[E: GridworldEnv]:

    env: E
    config: DictConfig
    agents: list[Agent]

    def __init__(self, env: E, config: DictConfig | dict | list) -> None:

        env.create_world()
        self.env = self.populate_environment(env)
        if isinstance(config, DictConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            # note that if config is a list, we assume it is a dotlist
            self.config = OmegaConf.from_dotlist(config)

        self.agents = self.setup_agents()

    @abstractmethod
    @staticmethod
    def setup_agents(self) -> list[Agent]:
        pass

    @abstractmethod
    @staticmethod
    def populate_environment(self, env: E) -> E:
        pass

    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents."""
        self.env.create_world()
        self.env = self.populate_environment(self.env)
        for agent in self.agents:
            agent.reset()

    def run(
        self, animate: bool = True, logging: bool = True, logger: Logger | None = None
    ) -> None:
        """Run the experiment.

        If animate is true,
        animates the experiment every record_period (determined by `self.config.experiment.record_period`).

        If logging is true, logs the total loss and total rewards each epoch.

        Args:
            animate: Whether to animate the experiment. Defaults to True.
            logging: Whether to log the experiment. Defaults to True.
            logger: The logger to use. Defaults to a ConsoleLogger.
        """
        imgs = []
        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # start epoch action for each agent model
            for agent in self.agents:
                agent.model.start_epoch_action(**locals())

            # run the environment for the specified number of turns
            while not self.env.turn >= self.config.experiment.max_turns:
                if animate and epoch % self.config.experiment.record_period == 0:
                    full_sprite = render_sprite(self.env)
                    imgs.append(image_from_array(full_sprite))

                self.env.take_turn()

            # generate the gif if animate is true
            if animate and epoch % self.config.experiment.record_period == 0:
                animate(
                    imgs,
                    f"{self.env.__name__}_epoch{epoch}",
                    Path(__file__).parent / "./data/",
                )
                imgs = []

            # At the end of each epoch, train the agents.
            total_loss = 0
            for agent in self.agents:
                loss = agent.model.train_step()
                total_loss += loss

            if logging:
                if not logger:
                    logger = ConsoleLogger(self.config.experiment.epochs)
                # TODO: implement env game_score
                logger.record_turn(
                    epoch, total_loss, self.env.game_score, self.agents[0].model.epsilon
                )

            # update epsilon
            for agent in self.agents:
                agent.model.epsilon_decay(self.config.experiment.epsilon_decay)
