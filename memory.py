import random
from collections import deque


class Experience:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def __str__(self):
        return f"State:\n{self.state}\nAction: {self.action}\nReward: {self.reward}\nNext State:\n{self.next_state}\n\n"


class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        output_string = ""
        for exp in self.memory:
            output_string += str(exp)
        return output_string

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
