from cart_pole.model import Actor
import tensorflow as tf
import gym
import numpy as np
import math


class Trainer:
    def __init__(self, actor=None):
        self.env = gym.make('CartPole-v0')
        self.actor = Actor(model=actor) if actor else \
            Actor.init_model(1, self.env.observation_space.shape[0], 100, 2)
        self.episode_length = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.variables = self.actor.model.trainable_variables
        self.trainable_weights = self.actor.model.trainable_weights
        self.discount_factor = 0.99

    def reset(self):
        self.epsiode_length = 0

    def record_episode(self, iterations):
        done = False
        inital_state = self.env.reset()
        states = np.ones([iterations, self.env.observation_space.shape[0]])
        states[0] = inital_state
        rewards = np.zeros([iterations])
        actions = np.ones([iterations, self.env.action_space.n])
        gradients = []

        for i in range(iterations-1):
            if done:
                break
            self.episode_length = i
            with tf.GradientTape() as tape:
                policy = self.actor.get_policy(states[i])
                action = tf.random.categorical(policy, num_samples=1) \
                    .numpy()[0][0]
                action_log_prob = tf.math.log(policy[:, action] + 1e-50)

            gradients.append(tape.gradient(action_log_prob, self.variables))
            state, rewards[i], done, _ = self.env.step(action)
            states[i+1] = state
            actions[i] = action

        self.env.close()
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.gradients = gradients
        return states, rewards, actions, gradients

    def train(self):
        rewards = self.rewards[:self.episode_length]
        gradients = self.gradients[:self.episode_length]
        discounted_rewards, mean_score = self.discount_rewards(rewards)
        gradients = gradients*discounted_rewards[:, None]
        for grad in gradients:
            self.opt.apply_gradients(zip(grad, self.trainable_weights))
        return mean_score

    def discount_rewards(self, rewards):
        for i, reward in enumerate(rewards):
            pow = len(rewards) - 1 - i
            rewards[i] = math.pow(self.discount_factor, pow) * - 1 * reward
        mean = rewards.mean()
        return rewards - mean, mean
