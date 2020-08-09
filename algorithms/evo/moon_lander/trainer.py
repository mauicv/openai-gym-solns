"""Implements policy gradient method."""

from algorithms.model import ContinuousActor
from algorithms.evo.moon_lander.memory import Memory

import tensorflow as tf
import gym
# import numpy as np


class Trainer:
    def __init__(self, burn_in_eps=30, actor=None):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.memory = Memory(max_size=100)
        self.burn_in_eps = burn_in_eps
        self.eps = 0
        self.actions_dim = self.env.action_space.shape[0]
        self.episode_length = 0
        self.actor_learning_rate = 0.00001
        self.exploration_value = 1

        self.actor = ContinuousActor(model=actor) if actor else \
            ContinuousActor.init_model(2, self.env.observation_space.shape[0],
                                       256, self.env.action_space.shape[0])
        adamOpt = tf.keras.optimizers\
            .Adam(learning_rate=self.actor_learning_rate)
        self.actor.model.compile(optimizer=adamOpt,
                                 loss="mse",
                                 metrics=["mae"])

    def run_episode(self):
        done = False
        self.eps = self.eps + 1
        self.episode_length = 0
        state = self.env.reset()
        reward_sum = 0

        states = []
        actions = []

        while not done:
            self.episode_length += 1
            action = self.actor.get_action(state)
            action = action + tf.random\
                .normal([2], mean=0.0,
                        stddev=self.exploration_value,
                        dtype=tf.dtypes.float64)
            next_state, reward, done, _ = self.env.step(action)
            reward_sum = reward_sum + reward
            states.append(state)
            actions.append(action)

        self.memory.remember(states, actions, reward_sum)
        self.env.close()

        self.train()
        return self.memory.average_score()

    def train(self):
        if self.episode_length > self.burn_in_eps:
            states, actions = self.memory.sample()
            self.actor.model.fit(states, actions)
