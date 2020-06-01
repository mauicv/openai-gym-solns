"""Implements policy gradient method."""

from algorithms._moon_lander.model import Actor
import tensorflow as tf
import gym
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, actor=None):
        self.env = gym.make('LunarLander-v2')
        self.actor = Actor(model=actor) if actor else \
            Actor.init_model(4, self.env.observation_space.shape[0], 100, 4)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.variables = self.actor.model.trainable_variables
        self.discount_factor = 0.99
        self.e = 0.0
        self.orbits = OrbitsStore(10)

        self.states = []
        self.rewards = []
        self.gradients = []
        self.actions = []

    def reset(self):
        self.orbits.add(self.states)
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
        self.opt.apply_gradients(zip(mean_gradient, self.variables))
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

        current_orbit = self.orbits.match_orbit_lengths(self.states)
        weighted_path_var = self.orbits.compute_weighting(current_orbit)
        weighted_path_var, self.rewards = \
            pad_to_match(weighted_path_var, self.rewards)
        self.rewards = self.rewards*weighted_path_var
        return self.rewards, reward_sum


def pad_to_match(arr_1, arr_2):
    if len(arr_1) > len(arr_2):
        padding = (0, len(arr_1) - len(arr_2))
        arr_2 = np.pad(arr_2, padding, mode='edge')
    else:
        padding = (0, len(arr_2) - len(arr_1))
        arr_1 = np.pad(arr_1, padding, mode='edge')
    return arr_1, arr_2


class OrbitsStore(object):
    """Contains list of previous orbits.

    Used to compute rolling averages and variances in path seperation.
    """
    def __init__(self, num_orbits):
        self.orbits = deque([])
        self.num_orbits = num_orbits
        self.max_orb_len = 0
        self.old = None
        self.new = None
        self.avg_orbit = None

    def match_orbit_lengths(self, new):
        if self.max_orb_len > len(new):
            padding = ((0, self.max_orb_len - len(new)), (0, 0))
            new = np.pad(new, padding, mode='edge')
        else:
            for i in range(len(self.orbits)):
                padding = ((0, len(new) - self.max_orb_len), (0, 0))
                self.orbits[i] = np.pad(self.orbits[i], padding, mode='edge')
            self.max_orb_len = len(new)
            new = np.array(new)
        return new

    def add(self, new):
        new = self.match_orbit_lengths(new)

        if len(self.orbits) > self.num_orbits:
            old = self.orbits.popleft()
            self.orbits.append(new)
        else:
            self.orbits.append(new)
        self.avg_orbit = self.compute_avg_orbit()

    def avg_dists(self):
        states = np.stack(self.orbits, axis=0)
        mean_diff_vects = states - self.avg_orbit[None, :, :]
        mean_dist = np.mean(np.sqrt(np.sum(mean_diff_vects**2, axis=2)),
                            axis=0)
        std_dist = np.std(np.sqrt(np.sum(mean_diff_vects**2, axis=2)),
                          axis=0)
        return mean_dist, std_dist

    def compute_avg_orbit(self):
        states = np.stack(self.orbits, axis=0)
        avg_orb = np.mean(states, axis=0)
        return avg_orb

    def compare(self, current_orbit):
        if not np.any(self.avg_orbit):
            return np.ones((len(current_orbit), 8)), np.ones(len(current_orbit))

        current_orbit = np.array(current_orbit)
        diffs = current_orbit - self.avg_orbit
        mean_dist = np.sqrt(np.sum(diffs**2, axis=1))
        significance = np.sqrt(np.sum(mean_dist))
        return mean_dist, significance

    def compute_weighting(self, current_orbit):
        if not np.any(self.avg_orbit):
            return np.ones(len(current_orbit) - 1)

        self.avg_orbit = self.compute_avg_orbit()
        _, sig = self.compare(current_orbit)
        _, path_var = self.avg_dists()

        max_path_var = 1 if np.max(path_var) == 0.0 else np.max(path_var)
        print(max_path_var)
        path_var = path_var/max_path_var
        weighted_path_var = path_var/sig if sig > 1 else path_var
        return weighted_path_var
