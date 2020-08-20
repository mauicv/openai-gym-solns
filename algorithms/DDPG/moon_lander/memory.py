from collections import deque
import numpy as np
import random


class Memory:
    def __init__(self, batch_size, max_size=1000000):
        self.deque = deque()
        self.batch_size = batch_size
        self.max_size = max_size

    def remember(self, state, action, reward, done, next_state):
        self.deque.append((state, action, reward, done, next_state))
        if len(self.deque) > self.max_size:
            self.deque.popleft()

    def sample(self):
        sample = random.sample(self.deque, self.batch_size)
        states, actions, rewards, done, next_states = zip(*sample)
        return np.array(states), np.array(actions), np.array(rewards), \
            np.array(done), np.array(next_states)

    def full(self):
        return len(self.deque) > self.batch_size

    def get_avg(self):
        sum_of_reward = 0
        for item in self.deque:
            sum_of_reward = sum_of_reward + item[2]
        return sum_of_reward/len(self.deque)

    def check(self):
        actions = np.array([r[1] for r in list(self.deque)[-200:]])
        print('    Average action value: ', np.abs(actions).mean())
