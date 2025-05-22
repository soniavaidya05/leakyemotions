from abc import abstractmethod
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from sorrel.agents import Agent
from sorrel.environments import GridworldEnv
from sorrel.utils.logging import ConsoleLogger, Logger

# TODO: change animate to animate_gif
from sorrel.utils.visualization import (
    ImageRenderer,
    animate_gif,
    image_from_array,
    render_sprite,
)


class Experiment[E: GridworldEnv]:

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
        pass

    @abstractmethod
    def populate_environment(self) -> None:
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

        If animate is true,
        animates the experiment every record_period (determined by `self.config.experiment.record_period`).

        If logging is true, logs the total loss and total rewards each epoch.

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
