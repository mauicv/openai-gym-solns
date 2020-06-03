"""Implements policy gradient method."""

from algorithms.model import Critic
import tensorflow as tf
import gym
import numpy as np


class Trainer:
    def __init__(self, critic=None):
        self.env = gym.make('CartPole-v0')
        self.Q = Critic(model=critic) if critic else \
            Critic.init_model(2, self.env.observation_space.shape[0]
                              + self.env.action_space.n, 64)

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actions_dim = self.env.action_space.n
        self.variables = self.Q.model.trainable_variables
        self.discount_factor = 0.99
        self.e = 0.01
        self.episode_length = 0

        self.states = []
        self.rewards = []
        self.gradients = []
        self.actions = []

    def record_episode(self, iterations):
        done = False
        self.episode_length = 0
        state = self.env.reset()
        iterations = 200 if iterations is None else iterations

        while not done or self.episode_length > iterations:
            self.episode_length += 1

            with tf.GradientTape() as tape:
                action_Qs = self.get_action_vals(state)
                action = np.argmax(action_Qs)
                Q_estimate = action_Qs[action]

            grads = tape.gradient(Q_estimate, self.variables)
            state, reward, done, _ = self.env.step(action)
            future_Q = max(self.get_action_vals(state))
            delta = reward + self.discount_factor * future_Q - Q_estimate
            for grad in grads:
                grad = grad*delta
            self.opt.apply_gradients(zip(grads, self.variables))

        self.env.close()
        return self.episode_length

    def get_action_vals(self, state):
        Q = []
        all_actions_one_hot = tf.one_hot(list(range(self.actions_dim)),
                                         self.actions_dim)
        for one_hot_action in all_actions_one_hot:
            Q_a = self.Q.get_Q(np.array(state), one_hot_action)
            Q.append(Q_a)
        return Q
