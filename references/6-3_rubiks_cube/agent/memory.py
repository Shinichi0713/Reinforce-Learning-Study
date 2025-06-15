import numpy as np


class Memory:
    def __init__(self, mem_limit=1024, batch_size=128):
        self.limit = mem_limit
        self.batch_size = batch_size

        # pointing index to be filled next
        self.head = 0
        self.memory = [None] * self.limit
        self.full = False

    # FIFO
    def push(self, state, action, reward, next_state, done):
        self.memory[self.head] = (state, action, reward, next_state, done)
        self.head = (self.head + 1) % self.limit
        if self.head == 0:
            self.full = True

    def sample(self):
        # NOTE: np.random.randint() samples from [low, high).
        # _keys = np.random.randint(0, self._get_len(), self.batch_size)
        _keys = np.random.choice(range(self._get_len()),
                                 self.batch_size, replace=False)
        data = [self.memory[i] for i in _keys]
        return data

    def get_memory_data(self):
        return self.memory[:self._get_len()]

    def is_full(self):
        return self.full

    def reset(self):
        self.head = 0
        self.memory = [None] * self.limit
        self.full = False

    def _get_len(self):
        if self.full:
            return self.limit
        else:
            return self.head
