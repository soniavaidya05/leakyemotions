# general imports
import torch

from agentarium.models.pytorch import PyTorchIQN
from agentarium.observation.observation import Observation
# agentarium imports
from agentarium.utils.logging import GameLogger
from agentarium.utils.visualization import (animate, image_from_array,
                                            visual_field_sprite)
from examples.treasurehunt.agents import TreasurehuntAgent
# imports from our example
from examples.treasurehunt.env import Treasurehunt

# experiment parameters
EPOCHS = 1000
MAX_TURNS = 100
EPSILON_DECAY = 0.001
ENTITY_LIST = ["EmptyEntity", "Wall", "Gem", "TreasurehuntAgent"]
RECORD_PERIOD = 50  # how many epochs in each data recording period


def setup() -> Treasurehunt:
    """Set up all the whole environment and everything within."""
    # world configurations
    height = 7
    width = 7
    gem_value = 10
    spawn_prob = 0.002

    # make the agents
    agent_num = 2
    agents = []
    for _ in range(agent_num):
        observation = Observation(ENTITY_LIST)

        model = PyTorchIQN(
            input_size=(len(ENTITY_LIST), height, width),
            action_space=4,
            layer_size=250,
            epsilon=0.7,
            device="cpu",
            seed=torch.random.seed(),
            num_frames=5,
            n_step=3,
            sync_freq=200,
            model_update_freq=4,
            BATCH_SIZE=64,
            memory_size=1024,
            LR=0.00025,
            TAU=0.001,
            GAMMA=0.99,
            N=12,
        )

        agents.append(TreasurehuntAgent(observation, model))

    # make the environment
    env = Treasurehunt(height, width, gem_value, spawn_prob, MAX_TURNS, agents)
    return env


def run(env: Treasurehunt):
    """Run the experiment."""
    imgs = []
    total_score = 0
    total_loss = 0
    for epoch in range(EPOCHS):
        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            if epoch % RECORD_PERIOD == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

        # TODO: delete this line
        assert all(agent.isdone(env) for agent in env.agents)

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in env.agents:
                loss = agent.model.train_model()
                total_loss += loss

        # Special action: update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - EPSILON_DECAY
            agent.model.epsilon = max(new_epsilon, 0.01)

        total_score += env.game_score

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / RECORD_PERIOD
            print(
                f"Epoch: {epoch}; Epsilon: {env.agents[0].model.epsilon}; Losses this period: {total_loss}; Avg. score this period: {avg_score}"
            )
            # TODO: docs: need to create a /data folder to have the sprites save properly!
            animate(imgs, f"treasurehunt_epoch{epoch}", "../data/")

            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0


if __name__ == "__main__":
    env = setup()
    run(env)
