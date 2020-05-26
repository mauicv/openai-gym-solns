"""Implements policy gradient method."""

from moon_lander.model import Actor
import tensorflow as tf
import gym
import numpy as np
import math


class Trainer:
    def __init__(self, actor=None):
        self.env = gym.make('LunarLander-v2')
        self.actor = Actor(model=actor) if actor else \
            Actor.init_model(4, self.env.observation_space.shape[0], 100, 4)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.variables = self.actor.model.trainable_variables
        self.discount_factor = 0.99
        self.e = 0.0

        self.states = []
        self.rewards = []
        self.gradients = []
        self.actions = []

    def reset(self):
        self.epsiode_length = 0
        self.states = []
        self.rewards = []
        self.gradients = []
        self.actions = []

    def record_episode(self, num_steps):
        done = False
        inital_state = self.env.reset()
        self.states.append(inital_state)
        i = 0
        while not done:
            if num_steps and i >= num_steps:
                break
            with tf.GradientTape() as tape:
                policy = self.actor.get_policy(self.states[i])
                action = self.sample_action(policy)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=policy, labels=np.array([action]))
            grads = tape.gradient(loss, self.variables)
            self.gradients.append(grads)
            state, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            self.states.append(state)
            self.actions.append(action)
            i += 1
        self.env.close()

    def sample_action(self, policy):
        if np.random.rand(1) < self.e:
            action = self.env.action_space.sample()
        else:
            soft_max_prob = tf.nn.softmax(policy)
            action = np.random.choice(range(self.env.action_space.n),
                                      p=soft_max_prob.numpy()[0])
        return action

    def train(self):
        discounted_rewards, mean_score = self.discount_rewards()
        grad_sum = self.gradients[0]
        for i in range(len(self.gradients)):
            if i > 0:
                norm_grad = self.gradients[i]*discounted_rewards[i, None]
                grad_sum = grad_sum + norm_grad
        mean_gradient = grad_sum/len(self.gradients)
        self.opt.apply_gradients(
            zip(mean_gradient, self.variables))
        return mean_score

    def discount_rewards(self):
        self.rewards = np.array(self.rewards)
        reward_sum = sum(self.rewards)
        rewards = np.zeros(len(self.rewards))
        for i, currrent_reward in enumerate(self.rewards):
            future_discounted_rewards = [
                math.pow(self.discount_factor, j) * reward
                for j, reward in enumerate(self.rewards[i:])]
            rewards[i] = sum(future_discounted_rewards)
        mean = rewards.mean()
        rewards = rewards - mean
        std = np.std(rewards) if np.std(rewards) > 0 else 1
        self.rewards = rewards/std
        # print(self.rewards)
        return self.rewards, reward_sum
