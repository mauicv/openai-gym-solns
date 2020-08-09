"""Implements policy gradient method."""

from algorithms.model import ContinuousActor, Critic
from algorithms.DDPG.moon_lander.memory import Memory

import tensorflow as tf
import gym
# import numpy as np


class Trainer:
    def __init__(self, tau=0.05, burn_in_eps=30, critic=None, actor=None):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.memory = Memory(batch_size=50)
        self.tau = tau
        self.burn_in_eps = burn_in_eps
        self.eps = 0
        self.actions_dim = self.env.action_space.shape[0]
        self.discount_factor = 0.99
        self.episode_length = 0
        self.actor_learning_rate = 0.00001
        self.critic_learning_rate = 0.00001
        self.exploration_value = 0.2

        self.actor = ContinuousActor(model=actor) if actor else \
            ContinuousActor.init_model(2, self.env.observation_space.shape[0],
                                       400, self.env.action_space.shape[0])

        self.critic = Critic(model=critic) if critic else \
            Critic.init_model(2, self.env.observation_space.shape[0]
                              + self.env.action_space.shape[0], 400)

        if actor:
            self.target_actor = ContinuousActor(model=actor)
        else:
            self.target_actor = ContinuousActor.init_model(
                2, self.env.observation_space.shape[0],
                400, self.env.action_space.shape[0])
            self.target_actor.model.set_weights(self.actor.model.get_weights())

        if critic:
            self.target_critic = Critic(model=critic)
        else:
            self.target_critic = Critic.init_model(
                2, self.env.observation_space.shape[0]
                + self.env.action_space.shape[0], 400)
            self.target_critic.model\
                .set_weights(self.critic.model.get_weights())

        self.actor_variables = self.actor.model.trainable_variables
        self.critic_variables = self.critic.model.trainable_variables

        self.actor_opt = tf.keras.optimizers\
            .Adam(learning_rate=self.actor_learning_rate)
        self.critic_opt = tf.keras.optimizers\
            .Adam(learning_rate=self.critic_learning_rate)

    def run_episode(self):
        done = False
        self.eps = self.eps + 1
        self.episode_length = 0
        state = self.env.reset()
        reward_sum = 0
        success_count = 0
        while not done:
            self.episode_length += 1
            action = self.actor.get_action(state)
            action = action + tf.random\
                .normal([2], mean=0.0,
                        stddev=self.exploration_value,
                        dtype=tf.dtypes.float64)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.remember(state, action, reward, done, next_state)

            reward_sum = reward_sum + reward
            success = self.train()
            if success:
                success_count = success_count + 1
            state = next_state

        self.env.close()
        self.memory.check()
        return reward_sum/self.episode_length, \
            success_count/self.episode_length, \
            self.episode_length

    def train(self):
        success = True
        if self.memory.full():
            success = False
            states, actions, rewards, done, next_states = self.memory.sample()

            with tf.GradientTape() as actor_tape, \
                    tf.GradientTape() as critic_tape:
                Q_loss = self\
                    .Q_loss(states, actions, rewards, next_states, done)
                if self.eps > self.burn_in_eps:
                    action_loss = tf.math.negative(self.action_loss(states))

            if self.eps > self.burn_in_eps:
                actor_grads = actor_tape\
                    .gradient(action_loss, self.actor_variables)
                self.actor_opt \
                    .apply_gradients(zip(actor_grads, self.actor_variables))

            critic_grads = critic_tape\
                .gradient(Q_loss, self.critic_variables)
            self.critic_opt \
                .apply_gradients(zip(critic_grads,
                                 self.critic_variables))

            if self.eps > self.burn_in_eps:
                updated_action_loss = tf.math\
                    .negative(self.action_loss(states)).numpy()
                if updated_action_loss > action_loss:
                    success = True

            self.target_actor.track_weights(self.tau, self.actor.model)
            self.target_critic.track_weights(self.tau, self.critic.model)
        return success

    def Q_loss(self, states, actions, rewards, next_states, done):
        next_actions = self.target_actor.model(next_states)
        Q_input = tf.concat([next_states, next_actions], axis=1)
        y = rewards[:, None] + self.discount_factor*(1-done)*self\
            .target_critic.model(Q_input)
        Q_input = tf.concat([states, actions], axis=1)
        squared_error = tf.pow(y - self.critic.model(Q_input), 2)
        return tf.reduce_mean(squared_error)

    def action_loss(self, states):
        actions = self.actor.model(states)
        Q_input = tf.concat([states, actions], axis=1)
        mean = tf.reduce_mean(self.critic.model(Q_input))
        return mean
