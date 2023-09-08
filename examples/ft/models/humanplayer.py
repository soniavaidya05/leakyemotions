from matplotlib import pyplot as plt
import numpy as np
from IPython.display import clear_output

class ModelHumanPlayer:

    def __init__(self, action_space, memory_size):
        self.action_space = action_space
        self.memory_size = memory_size

    def take_action(self, state):
        
        clear_output(wait = True)
        for i in range(self.memory_size):
            frame = state[:, i, :, :, :].squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            plt.subplot(1, self.memory_size, i+1)
            plt.imshow(frame)
        plt.show()
        
        done = 0
        while done == 0:
            action = int(input("Select Action: "))
            if action in self.action_space:
                done = 1
            else:
                print("Please try again. Possible actions are below.")
                print(self.actionSpace)
                # we can have iinputType above also be joystick, or other controller

        return action


