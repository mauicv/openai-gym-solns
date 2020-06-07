"""Implements policy gradient method."""

from algorithms.model import Actor, Critic
import tensorflow as tf
import gym
import numpy as np


class Trainer:
    def __init__(self, critic=None, actor=None):
        self.env = gym.make('CartPole-v0')

        self.actor = Actor(model=actor) if critic else \
            Actor.init_model(2, self.env.observation_space.shape[0],
                             64, self.env.action_space.n)

        self.critic = Critic(model=critic) if critic else \
            Critic.init_model(2, self.env.observation_space.shape[0]
                              + self.env.action_space.n, 64)

        self.actor_variables = self.actor.model.trainable_variables
        self.critic_variables = self.critic.model.trainable_variables

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.actions_dim = self.env.action_space.n
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

            with tf.GradientTape() as actor_tape, \
                    tf.GradientTape() as critic_tape:
                policy = self.actor.get_policy(state)
                action = self.sample_action(policy)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=policy, labels=np.array([action]))

                q_val = self.get_action_q_value(state, action)

            actor_grads = actor_tape.gradient(loss, self.actor_variables)
            critic_grads = critic_tape.gradient(q_val, self.critic_variables)

            state, reward, done, _ = self.env.step(action)

            # first update the actor model with the q_val as weight.
            actor_grads = [grad*q_val[0] for grad in actor_grads]
            self.actor_opt \
                .apply_gradients(zip(actor_grads, self.actor_variables))

            # second update the critic model with the target val:
            # reward = 0.01 if not done else -1
            future_Q = self.get_future_Q(state)
            target = reward + self.discount_factor * future_Q
            target_delta = q_val - target
            critic_grads = [grad*target_delta[0] for grad in critic_grads]
            self.critic_opt \
                .apply_gradients(zip(critic_grads, self.critic_variables))

            # print('--------------------------------------')
            # print('done: ', done)
            # print('reward: ', reward)
            # print('delta:', delta.numpy()[0][0])
            # print('Q_estimate:', Q_estimate.numpy()[0][0])
            # print('future_Q:', future_Q.numpy()[0][0])

        self.env.close()
        return self.episode_length

    def sample_action(self, policy):
        if np.random.rand(1) < self.e:
            action = self.env.action_space.sample()
        else:
            soft_max_prob = tf.nn.softmax(policy)
            action = np.random \
                .choice([0, 1], p=soft_max_prob.numpy()[0])
        return action

    def get_action_q_value(self, state, action):
        one_hot_action = tf.one_hot(action, self.actions_dim)
        return self.critic.get_Q(np.array(state), one_hot_action)

    def get_future_Q(self, state):
        action_Qs = []
        all_actions_one_hot = tf.one_hot(list(range(self.actions_dim)),
                                         self.actions_dim)
        for one_hot_action in all_actions_one_hot:
            Q_a = self.critic.get_Q(np.array(state), one_hot_action)
            action_Qs.append(Q_a)
        return max(action_Qs)
