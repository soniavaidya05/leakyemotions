"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# TODO: 3rd file to write!

import numpy as np

from agentarium.models.pytorch import PyTorchIQN
from agentarium.primitives import Agent, GridworldEnv


class TreasurehuntAgent(Agent):
    """
    A treasurehunt agent that uses the iqn model.
    """

    def __init__(self, model):
        action_space = [0, 1, 2, 3]  # up, down, left, right
        super().__init__(model, action_space)

        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/hero.png"
        )

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        # TODO
        pass

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states"""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment."""

        new_location = tuple()
        if action == 0:  # UP
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:  # DOWN
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:  # LEFT
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:  # RIGHT
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = env.observe(new_location)
        reward = target_object.value

        # try moving to new_location
        env.move(self, new_location)

        return reward

    def is_done(self, env: GridworldEnv) -> bool:
        return False
