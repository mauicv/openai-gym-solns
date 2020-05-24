from cart_pole.model import Actor
import tensorflow as tf
import gym
import numpy as np
import math


class Trainer:
    def __init__(self, actor=None):
        print(actor)
        self.env = gym.make('CartPole-v0')
        self.actor = Actor(model=actor) if actor else \
            Actor.init_model(2, self.env.observation_space.shape[0], 64, 2)
        self.episode_length = 0
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.variables = self.actor.model.trainable_variables
        self.discount_factor = 0.99
        self.e = 0.01

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
                if np.random.rand(1) < self.e:
                    action = self.env.action_space.sample()
                else:
                    soft_max_prob = tf.nn.softmax(policy)
                    action = np.random \
                        .choice([0, 1], p=soft_max_prob.numpy()[0])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=policy, labels=np.array([action]))
                # loss = tf.math.negative(loss)
            grads = tape.gradient(loss, self.variables)
            self.gradients.append(grads)
            state, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            self.states.append(state)
            self.actions.append(action)

        self.env.close()
        # self._display_actions()
        return self.states, self.rewards, self.actions, self.gradients

    def train(self):
        discounted_rewards, mean_score = self.discount_rewards()
        grad_sum = self.gradients[0]
        for i in range(len(self.gradients)):
            if i > 0:
                norm_grad = self.gradients[i]*discounted_rewards[i, None]
                grad_sum = grad_sum + norm_grad
        mean_gradient = grad_sum/len(self.gradients)

        # self._display_action_probs()
        self.opt.apply_gradients(
            zip(mean_gradient, self.variables))
        # self._display_action_probs()

        return len(self.rewards)

    def discount_rewards(self):
        self.rewards = np.array(self.rewards)
        for i, _ in enumerate(self.rewards):
            future_discounted_rewards = [
                math.pow(self.discount_factor, j) * reward
                for j, reward in enumerate(self.rewards[i:])]
            self.rewards[i] = sum(future_discounted_rewards)
        mean = self.rewards.mean()
        std = np.std(self.rewards) if np.std(self.rewards) > 0 else 1
        self.rewards = (self.rewards - mean)/std
        # self._display_rewards()
        return self.rewards, mean

    def _display_action_probs(self):
        ap_after = [tf.nn.softmax(self.actor.get_policy(s))[0][a].numpy()
                    for a, s in zip(self.actions, self.states)]
        print('action probs: ', ['%.4f' % r for r in ap_after])

    def _display_actions(self):
        print('actions     : ', ['%.4f' % r for r in self.actions])

    def _display_rewards(self):
        print('rewards     : ', ['%.4f' % r for r in self.rewards])
