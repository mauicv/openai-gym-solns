import numpy as np
# import random


class Memory:
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size

    def remember(self, states, actions, reward):
        if len(self.memory) < self.max_size:
            self.memory.append((np.array(states), np.array(actions), reward))
            return

        minimum = self.get_min()
        if reward > minimum:
            self.pop_min(minimum)
            self.memory.append((np.array(states), np.array(actions), reward))

    def sample(self):
        states = []
        actions = []
        for orbit in self.memory:
            states.extend(orbit[0])
            actions.extend(orbit[1])
        return np.array(states), np.array(actions)

    def pop_min(self, minimum):
        for index, memory in enumerate(self.memory):
            if memory[2] == minimum:
                self.memory.pop(index)

    def get_min(self):
        if len(self.memory) == 0:
            return -100000
        return min([orb[2] for orb in self.memory])

    def average_score(self):
        return sum([orb[2] for orb in self.memory])/len(self.memory)
