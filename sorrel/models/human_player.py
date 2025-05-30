from typing import Sequence

import numpy as np
from IPython.display import clear_output

from sorrel.buffers import Buffer
from sorrel.models import BaseModel
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.utils.visualization import plot, render_sprite
from sorrel.worlds import Gridworld

class HumanObservation(ObservationSpec[str]):

    def observe(
        self,
        world: Gridworld,
        location: tuple | None = None
    ) -> np.ndarray: 
        
        sprite = render_sprite(
            world=world,
            location=location,
        )

        print([s.shape for s in sprite])
        
        return np.array(sprite)
    
class HumanMemory(Buffer):
    def current_state(self) -> np.ndarray:
        return self.states[-1]
        

class HumanPlayer(BaseModel):
    """Model subclass for a human player."""

    def __init__(
        self,
        input_size: Sequence[int],
        action_space: int,
        memory_size: int,
        show: bool = True,
    ):
        self.name = ""
        self.action_list = np.arange(action_space)
        self.num_frames = memory_size
        self.show = show
        self.input_size = tuple([shape * 16 for shape in input_size])
        _input_size = (1, self.input_size[0] * self.input_size[1] * 3 * 4)
        self.SLICE = np.prod(_input_size)
        self.memory = HumanMemory(1, _input_size, 1)

    def take_action(self, state: np.ndarray):
        """Observe a visual field sprite output."""

        if self.show:
            # clear_output(wait=True)

            state = state[:, self.SLICE:]
            state = state.reshape(
                (self.input_size[0], self.input_size[1], 4, -1)
            )
            state = np.array(state, dtype=int)
            print(state.shape)
            print(state[:, :, :, 0])

            state_ = []
            for i in range(state.shape[3]):
                state_.append(
                    state[:, :, :, i]
                )
            plot(state_)

        action = None
        num_retries = 0
        while not isinstance(action, int):
            action_ = input("Select Action: ")
            if action_ in ["w", "a", "s", "d"]:
                if action_ == "w":
                    action = 0
                elif action_ == "s":
                    action = 1
                elif action_ == "a":
                    action = 2
                elif action_ == "d":
                    action = 3
            elif action_ in [str(act) for act in self.action_list]:
                action = int(action_)
            elif action == "quit":
                raise KeyboardInterrupt("Quitting...")
            else:
                num_retries += 1
                if num_retries > 5:
                    raise KeyboardInterrupt("Too many invalid inputs. Quitting...")
                print("Please try again. Possible actions are below.")
                print(self.action_list + np.array(["quit"]))

        return action
