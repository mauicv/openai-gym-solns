"""Simple environment for TESTING

The state is either 0 or 1, if the agent chooses the state value they are
given -1 if they don't they get 1. So the agent should be choosing the opposite
state value at each time step. The state changes randomly on each step.
"""

import random
import numpy as np


class Env():
    def __init__(self):
        self.evolve_state()

    def action_space(self):
        return np.array([0, 1])

    def step(self, action):
        reward = -1 if action == self.state else 1
        self.evolve_state()
        done = False
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.evolve_state()
        return self.state

    def evolve_state(self):
        self.state = np.array([random.choice([0, 1])])
