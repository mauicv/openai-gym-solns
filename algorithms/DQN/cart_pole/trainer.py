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

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
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

                state, reward, done, _ = self.env.step(action)

                reward = 1 if not done else -1
                future_Q = max(tf.stop_gradient(self.get_action_vals(state)))
                final_state = 0 if done else 1
                target = reward + self.discount_factor * future_Q * final_state
                q_loss = tf.math.pow(Q_estimate - target, 2)

            grads = tape.gradient(q_loss, self.variables)

            # print('--------------------------------------')
            # print('done: ', done)
            # print('reward: ', reward)
            # print('delta:', delta.numpy()[0][0])
            # print('Q_estimate:', Q_estimate.numpy()[0][0])
            # print('future_Q:', future_Q.numpy()[0][0])

            # grads = [grad*delta[0] for grad in grads]
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
