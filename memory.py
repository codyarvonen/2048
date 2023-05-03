from collections import deque

class Experience:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)