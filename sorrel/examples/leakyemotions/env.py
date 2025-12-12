# begin imports
# general imports
import numpy as np
import omegaconf
import os
import torch
from pathlib import Path

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.models.pytorch.iqn import iRainbowModel
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.logging import Logger, ConsoleLogger, TensorboardLogger
from sorrel.utils.visualization import ImageRenderer

# imports from our example
from sorrel.examples.leakyemotions.agents import LeakyEmotionsAgent, Wolf
from sorrel.examples.leakyemotions.custom_observation_spec import LeakyEmotionsObservationSpec, InteroceptiveObservationSpec, OtherOnlyObservationSpec, NoEmotionObservationSpec
from sorrel.examples.leakyemotions.entities import EmptyEntity, Bush, Wall, Grass
from sorrel.examples.leakyemotions.wolf_model import WolfModel
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld



# end imports
ENTITY_LIST = ["EmptyEntity", "Bush", "Wall", "Grass", "LeakyEmotionsAgent", "Wolf"]

# begin leakyemotions environment
class LeakyEmotionsEnv(Environment[LeakyEmotionsWorld]):
    """The experiment for Leaky Emotions."""

    def __init__(self, world: LeakyEmotionsWorld, config: dict | omegaconf.DictConfig) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.config.world.agents
        emotion_length = self.config.model.emotion_length
        agents = []
        bunnies = []
        wolves = []
        for i in range(agent_num):
            # create the observation spec
            entity_list = ENTITY_LIST
            match self.config.model.emotion_condition:
                case "full":
                    observation_spec_class = LeakyEmotionsObservationSpec
                case "self":
                    observation_spec_class = InteroceptiveObservationSpec
                case "other":
                    observation_spec_class = OtherOnlyObservationSpec
                # Default case: no emotions
                case _:
                    observation_spec_class = NoEmotionObservationSpec
                    
            observation_spec = observation_spec_class(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
                emotion_length=emotion_length
            )
            
            # Flatten input
            observation_spec.override_input_size(
                np.zeros(observation_spec.input_size, dtype=int).reshape(1, -1).shape
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = iRainbowModel(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.05,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12
            )

            if "checkpoint" in self.config.model:
                model.load(
                    Path(__file__).parent / f"./data/checkpoints/trial{self.config.model.checkpoint}_agent{i}.pkl"
                )

            bunny = LeakyEmotionsAgent(
                observation_spec=observation_spec,
                action_spec=action_spec,
                model=model,
            )

            bunnies.append(bunny)
            agents.append(bunny)
        
        wolf_num = self.config.world.wolves
        for _ in range(wolf_num):
            # create the observation spec
            entity_list = ENTITY_LIST
            observation_spec = LeakyEmotionsObservationSpec(
                entity_list,
                full_view=False,
                # note that here we require self.config to have the entry model.agent_vision_radius
                # don't forget to pass it in as part of config when creating this experiment!
                vision_radius=self.config.model.agent_vision_radius,
            )
            
            observation_spec.override_input_size(
                np.zeros(observation_spec.input_size, dtype=int).reshape(1, -1).shape
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])
            
            wolf = Wolf(
                    observation_spec=observation_spec, 
                    action_spec=action_spec, 
                    model=WolfModel(1, 4, 1)
                )

            wolves.append(wolf)
            agents.append(wolf)

        self.agents = agents
        self.bunnies = bunnies
        self.wolves = wolves

    def override_agents(self, agents: list[Agent]) -> None:
        """Override the current agent configuration with a list of new agents and resets
        the environment.

        Args:
            agents: A list of new agents
        """
        self.agents = agents

    def populate_environment(self):
        """
        Populate the leakyemotions world by creating walls, then randomly spawning the agents.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        if hasattr(self.config.world, "start_bush"):
            self.start_bush = self.config.world.start_bush
        else:
            self.start_bush = 0

        valid_agent_spawn_locations = []
        valid_bush_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if y in [0, self.world.height - 1] or x in [0, self.world.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom (first) layer, put grass there
                self.world.add(index, Grass(bush_lifespan=self.config.world.bush_lifespan))
                valid_bush_spawn_locations.append(index)
            elif z == 1: # if location is on third layer, rabbit agents and wolves can appear here 
                valid_agent_spawn_locations.append(index)

        # spawn the agents (rabbits)
        # using np.random.choice, we choose indices in valid_agent_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_agent_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_agent_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)

        bush_locations_indices = np.random.choice(
            len(valid_bush_spawn_locations), size=self.start_bush, replace=False
        )
        bush_locations = [valid_bush_spawn_locations[i] for i in bush_locations_indices]
        for loc in bush_locations:
            loc = tuple(loc)
            self.world.add(loc, Bush(lifespan=self.config.world.bush_lifespan))

    def reset(self) -> None:
        """Reset the experiment, including the environment and the agents."""
        self.turn = 0
        self.world.is_done = False
        self.world.dead_agents = []
        self.world.create_world()
        self.populate_environment()
        for agent in self.agents:
            agent.reset()

    def run_experiment(
        self,
        animate: bool = True,
        logging: bool = True,
        logger: Logger | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Run the experiment.

        Required config parameters:
            - experiment.epochs: The number of epochs to run the experiment for.
            - experiment.max_turns: The maximum number of turns each epoch.
            - (Only if `animate` is true) experiment.record_period: The time interval at which to record the experiment.
        Optional config parameters:
            - experiment.zero_emotion_after_training: If true, runs a frozen evaluation pass with the emotion layer zeroed out.
            - experiment.zero_emotion_eval_epochs: The number of evaluation epochs to run when zeroing the emotion layer.

        If `animate` is true,
        animates the experiment every `self.config.experiment.record_period` epochs.

        If `logging` is true, logs the total loss and total rewards each epoch.

        Args:
            animate: Whether to animate the experiment. Defaults to True.
            logging: Whether to log the experiment. Defaults to True.
            logger: The logger to use. Defaults to a ConsoleLogger.
            output_dir: The directory to save the animations to. Defaults to "./data/" (relative to current working directory).
        """
        
        renderer = None
        if output_dir is None:
            if hasattr(self.config.experiment, "output_dir"):
                output_dir = Path(self.config.experiment.output_dir)
            else:
                output_dir = Path(__file__).parent / "./data/"
            assert isinstance(output_dir, Path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if animate:
            renderer = ImageRenderer(
                experiment_name=self.world.__class__.__name__,
                record_period=self.config.experiment.record_period,
                num_turns=self.config.experiment.max_turns,
            )

        for epoch in range(self.config.experiment.epochs + 1):
            # Reset the environment at the start of each epoch
            self.reset()

            # Determine whether to animate this turn.
            animate_this_turn = animate and (
                epoch % self.config.experiment.record_period == 0
            )

            # start epoch action for each agent model
            for agent in self.agents:
                agent.model.start_epoch_action(epoch=epoch)

            bunnies_left = sum([agent.alive for agent in self.bunnies])

            # run the environment for the specified number of turns
            while not (self.turn >= self.config.experiment.max_turns) and (bunnies_left > 0):
                if animate_this_turn and renderer is not None:
                    renderer.add_image(self.world)
                self.take_turn()
                bunnies_left = sum([agent.alive for agent in self.bunnies])
            # At the end, get the number of bushes
            active_bushes = sum([
                entity.kind == "Bush" for entity in self.world.map.flat
            ])
            average_pairwise_distance = np.mean([
                min(Wolf.compute_taxicab_distance(agent1.location, [agent2.location for agent2 in self.agents if agent1 != agent2])) for agent1 in self.agents
            ])

            self.world.is_done = True

            # generate the gif if animation was done
            if animate_this_turn and renderer is not None:
                renderer.save_gif(epoch, output_dir / "./gifs/")

            # At the end of each epoch, train the agents.
            total_loss = 0
            for agent in self.agents:
                loss = agent.model.train_step()
                total_loss += loss
                agent.model.epsilon_decay(self.config.model.epsilon_decay)

            # Log the information
            if logging:
                if not logger:
                    logger = ConsoleLogger(
                        self.config.experiment.epoch, 
                        "active_bushes",
                        "pairwise_distance"
                    )
                logger.record_turn(
                    epoch,
                    total_loss,
                    self.world.total_reward,
                    self.agents[0].model.epsilon,
                    active_bushes=active_bushes,
                    pairwise_distance=average_pairwise_distance
                )
            

            # update epsilon
            if epoch % 500 == 0:
                for i, agent in enumerate(self.agents):
                    os.makedirs(output_dir / f"./checkpoints/", exist_ok=True)
                    agent.model.save(output_dir / f'./checkpoints/trial{epoch}_agent{i}.pkl')

        # Optional zero-emotion evaluation phase (models stay frozen).
        eval_requested = self.config.experiment.get(
            "zero_emotion_after_training", False
        )
        eval_epochs = self.config.experiment.get("zero_emotion_eval_epochs", 0)

        if eval_requested and eval_epochs > 0:
            print(
                f"Running zeroed-emotion evaluation for {eval_epochs} epoch(s) with frozen models."
            )
            for agent in self.agents:
                observation_spec = getattr(agent, "observation_spec", None)
                if isinstance(observation_spec, LeakyEmotionsObservationSpec):
                    observation_spec.zero_emotion_layer(True)
                if hasattr(agent.model, "eval"):
                    agent.model.eval() #type: ignore

            eval_start_epoch = self.config.experiment.epochs + 1
            for epoch_offset in range(eval_epochs):
                epoch_number = eval_start_epoch + epoch_offset
                self.reset()

                animate_this_turn = animate and (
                    epoch_number % self.config.experiment.record_period == 0
                )

                for agent in self.agents:
                    agent.model.start_epoch_action(epoch=epoch_number)

                bunnies_left = sum([agent.alive for agent in self.bunnies])
                while not (self.turn >= self.config.experiment.max_turns) and (bunnies_left > 0):
                    if animate_this_turn and renderer is not None:
                        renderer.add_image(self.world)
                    self.take_turn()
                    bunnies_left = sum([agent.alive for agent in self.bunnies])

                self.world.is_done = True

                # generate the gif if animation was done
                if animate_this_turn and renderer is not None:
                    renderer.save_gif(epoch_offset, output_dir / "./gifs/")

                if logging and logger:
                    logger.record_turn(
                        epoch_number,
                        float("nan"),
                        self.world.total_reward,
                        self.agents[0].model.epsilon,
                    )

            for agent in self.agents:
                observation_spec = getattr(agent, "observation_spec", None)
                if isinstance(observation_spec, LeakyEmotionsObservationSpec):
                    observation_spec.zero_emotion_layer(False)
