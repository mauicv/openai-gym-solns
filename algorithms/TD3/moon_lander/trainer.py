"""Implements policy gradient method."""

from algorithms.model import ContinuousActor, Critic
from algorithms.TD3.moon_lander.memory import Memory

import tensorflow as tf
import gym
# import numpy as np


class Trainer:
    def __init__(self, tau=0.005, burn_in_eps=30, critics=None, actor=None):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.memory = Memory(batch_size=120)
        self.tau = tau
        self.burn_in_eps = burn_in_eps
        self.eps = 0
        self.actions_dim = self.env.action_space.shape[0]
        self.discount_factor = 0.99
        self.episode_length = 0
        self.actor_learning_rate = 0.00001
        self.critic_learning_rate = 0.00001
        self.exploration_value = 0.2
        self.smoothing_var = 0.05
        self.clipping_val = 0.5
        self.low_action = -1
        self.high_action = 1
        self.policy_freq = 2

        self.actor = ContinuousActor(model=actor) if actor else \
            ContinuousActor.init_model(2, self.env.observation_space.shape[0],
                                       400, self.env.action_space.shape[0])

        self.critic_1 = Critic(model=critics[0]) if all(critics) else \
            Critic.init_model(2, self.env.observation_space.shape[0]
                              + self.env.action_space.shape[0], 400)

        self.critic_2 = Critic(model=critics[1]) if all(critics) else \
            Critic.init_model(2, self.env.observation_space.shape[0]
                              + self.env.action_space.shape[0], 400)

        if actor:
            self.target_actor = ContinuousActor(model=actor)
        else:
            self.target_actor = ContinuousActor.init_model(
                2, self.env.observation_space.shape[0],
                400, self.env.action_space.shape[0])
            self.target_actor.model.set_weights(self.actor.model.get_weights())

        if all(critics):
            self.target_critic_1 = Critic(model=critics[0])
            self.target_critic_2 = Critic(model=critics[1])
        else:
            self.target_critic_1 = Critic.init_model(
                2, self.env.observation_space.shape[0]
                + self.env.action_space.shape[0], 400)
            self.target_critic_1.model\
                .set_weights(self.critic_1.model.get_weights())

            self.target_critic_2 = Critic.init_model(
                2, self.env.observation_space.shape[0]
                + self.env.action_space.shape[0], 400)
            self.target_critic_2.model\
                .set_weights(self.critic_2.model.get_weights())

        self.actor_variables = self.actor.model.trainable_variables
        self.critic_1_variables = self.critic_1.model.trainable_variables
        self.critic_2_variables = self.critic_2.model.trainable_variables

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

        while not done:
            self.episode_length += 1
            action = self.actor.get_action(state)
            action = action + tf.random\
                .normal([2], mean=0.0,
                        stddev=self.exploration_value,
                        dtype=tf.dtypes.float64)
            action = tf.clip_by_value(action,
                                      clip_value_min=self.low_action,
                                      clip_value_max=self.high_action)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.remember(state, action, reward, done, next_state)

            reward_sum = reward_sum + reward
            self.train()
            state = next_state
        self.env.close()

        return reward_sum, self.episode_length

    def train(self):
        if self.memory.full():
            states, actions, rewards, done, next_states = self.memory.sample()

            with tf.GradientTape() as actor_tape, \
                    tf.GradientTape() as critic_tape_1, \
                    tf.GradientTape() as critic_tape_2:
                y = self.compute_target(states, actions, rewards,
                                        next_states, done)

                Q_input = tf.concat([states, actions], axis=1)
                squared_error_1 = tf.pow(y - self.critic_1.model(Q_input), 2)
                squared_error_2 = tf.pow(y - self.critic_2.model(Q_input), 2)
                Q_loss_1 = tf.reduce_mean(squared_error_1)
                Q_loss_2 = tf.reduce_mean(squared_error_2)

                if self.update_policy:
                    action_loss = tf.math.negative(self.action_loss(states))

            critic_1_grads = critic_tape_1\
                .gradient(Q_loss_1, self.critic_1_variables)
            critic_2_grads = critic_tape_2\
                .gradient(Q_loss_2, self.critic_2_variables)

            self.critic_opt \
                .apply_gradients(zip(critic_1_grads,
                                 self.critic_1_variables))
            self.critic_opt \
                .apply_gradients(zip(critic_2_grads,
                                 self.critic_2_variables))

            if self.update_policy:
                actor_grads = actor_tape\
                    .gradient(action_loss, self.actor_variables)
                self.actor_opt \
                    .apply_gradients(zip(actor_grads, self.actor_variables))

                self.target_actor.track_weights(self.tau, self.actor.model)
                self.target_critic_1 \
                    .track_weights(self.tau, self.critic_1.model)
                self.target_critic_2 \
                    .track_weights(self.tau, self.critic_2.model)

            if self.update_policy:
                return Q_loss_1.numpy(), Q_loss_2.numpy(), action_loss
            else:
                return Q_loss_1.numpy(), Q_loss_2.numpy(), 0
        return 0, 0, 0

    @property
    def update_policy(self):
        return self.eps > self.burn_in_eps and \
                self.episode_length % self.policy_freq

    def compute_target(self, states, actions, rewards, next_states, done):
        next_actions = self.target_actor.model(next_states)
        smoothing_noise = tf.random\
            .normal(actions.shape, mean=0.0,
                    stddev=self.smoothing_var,
                    dtype=tf.dtypes.float64)
        clipped_smoothing_noise = \
            tf.clip_by_value(smoothing_noise,
                             clip_value_min=-self.clipping_val,
                             clip_value_max=self.clipping_val)
        next_actions = tf.clip_by_value(
                         clipped_smoothing_noise + next_actions,
                         clip_value_min=self.low_action,
                         clip_value_max=self.high_action)

        Q_input = tf.concat([next_states, next_actions], axis=1)
        Q_1_val = self.target_critic_1.model(Q_input)
        Q_2_val = self.target_critic_2.model(Q_input)
        Q_val = tf.math.minimum(Q_1_val, Q_2_val)
        y = rewards[:, None] + self.discount_factor*(1-done)*Q_val
        return y

    def action_loss(self, states):
        actions = self.actor.model(states)
        Q_input = tf.concat([states, actions], axis=1)
        mean = tf.reduce_mean(self.critic_1.model(Q_input))
        return mean
