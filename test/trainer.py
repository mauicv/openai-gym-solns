from test.model import Actor
import tensorflow as tf
import numpy as np
from test.env import Env
# import gym
# import math


class Trainer:
    def __init__(self, actor=None):
        self.actor = \
            Actor.init_model(1, 1, 10, 2)
        self.episode_length = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.variables = self.actor.model.trainable_variables
        self.discount_factor = 0.99
        self.e = 0.1
        self.env = Env()

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

    def record_episode(self, iterations):
        done = False
        inital_state = self.env.reset()
        self.states.append(inital_state)

        for i in range(iterations-1):
            if done:
                break
            self.episode_length = i
            with tf.GradientTape() as tape:
                policy = self.actor.get_policy(self.states[i])
                # print('policy: ', policy.numpy())
                if np.random.rand(1) < self.e:
                    action = np.random.choice(self.env.action_space())
                else:
                    soft_max_prob = tf.nn.softmax(policy)
                    # print('prob: ', soft_max_prob.numpy())
                    action = np.random \
                        .choice([0, 1], p=soft_max_prob.numpy()[0])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=policy, labels=np.array([action]))
                # loss = tf.math.negative(loss)
                # print('action: ', action)
            grads = tape.gradient(loss, self.variables)
            self.gradients.append(grads)
            state, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            self.states.append(state)
            self.actions.append(action)

            # print('--------------------------')
            # for grad in grads:
            #     # print(grad.name)
            #     print(grad.numpy())
            # print('--------------------------')

        # print('actions taken: ', self.actions)
        return self.states, self.rewards, self.actions, self.gradients

    def train(self):
        discounted_rewards, mean_score = self.discount_rewards()
        self.gradients = self.gradients*discounted_rewards[:, None]
        for index, grad in enumerate(self.gradients):
            # check change make high reward actions more likely!
            pred = tf.nn.softmax(
                self.actor.get_policy(self.states[index])).numpy()
            selected_action = self.actions[index]
            prob_action = pred[0][selected_action]

            self.opt.apply_gradients(zip(grad, self.variables))

            pred = tf.nn.softmax(
                self.actor.get_policy(self.states[index])).numpy()

            if self.rewards[index] < 0:
                print('outcome: ', pred[0][selected_action] < prob_action)
            else:
                print('outcome: ', pred[0][selected_action] > prob_action)

            # self._print_act_pred(index, False)
        return len(self.rewards)

    def discount_rewards(self):
        return np.array(self.rewards), None

    def _print_act_pred(self, index, prior=True):
        # ----------------------- FOR - TESTING -----------------------
        if prior:
            print('---------------------------------------------------')
        print('prediction prob {} learning:'
              .format('prior to' if prior else 'after'))
        pred = tf.nn.softmax(
            self.actor.get_policy(self.states[index])).numpy()
        selected_action = self.actions[index]
        print('     reward: ', self.rewards[index])
        print('     Action: ', selected_action)
        print('     From: ', pred)
        print('\n')
        if not prior:
            print('---------------------------------------------------')
        print('\n')
