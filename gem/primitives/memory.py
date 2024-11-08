from typing import Optional


class Memory:
    """
    Abstract agent class.
    """

    def __init__(self, memory_size: int):
        self.priorities = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextstates = []
        self.dones = []

    def clear(self):
        del self.priorities[:]
        del self.actions[:]
        del self.states[:]
        del self.nextstates[:]
        del self.rewards[:]
        del self.dones[:]

    def append(self, exp: tuple):
        '''
        Add an experience to the agent's memory.

        Parameters:
            exp: The tuple to add
        '''
        # Unpack
        priority, exp1 = exp
        state, action, reward, nextstate, done = exp1
        # Add to replay
        self.priorities.append(priority)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.nextstates.append(nextstate)
        self.dones.append(done)

    def get_last_memory(self, attr: Optional[str] = None):
        '''
        Get the latest memory from the replay.

        Parameters:
            attr: (Optional) the attribute to get.
            If not specified, it returns all elements
        '''
        if attr is None:
            return (
                self.priorities[-1],
                self.states[-1],
                self.actions[-1],
                self.rewards[-1],
                self.nextstates[-1],
                self.dones[-1]
            )
        else:
            return getattr(self, attr)[-1]
