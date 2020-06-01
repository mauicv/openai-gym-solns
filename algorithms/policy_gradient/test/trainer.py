from algorithms.model import Actor
from algorithms.policy_gradient.test.env import Env
import tensorflow as tf
import numpy as np


class Trainer:
    def __init__(self, actor=None):
        self.actor = Actor.init_model(1, 1, 10, 2)
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
                if np.random.rand(1) < self.e:
                    action = np.random.choice(self.env.action_space())
                else:
                    soft_max_prob = tf.nn.softmax(policy)
                    action = np.random \
                        .choice([0, 1], p=soft_max_prob.numpy()[0])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=policy, labels=np.array([action]))
            grads = tape.gradient(loss, self.variables)
            self.gradients.append(grads)
            state, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            self.states.append(state)
            self.actions.append(action)

        print('------------------------------------------------')
        self._display_states()
        self._display_actions()
        return self.states, self.rewards, self.actions, self.gradients

    def train(self):
        discounted_rewards, mean_score = self.discount_rewards()
        self.gradients = self.gradients*discounted_rewards[:, None]
        self._display_action_probs()
        for index, grad in enumerate(self.gradients):
            self.opt.apply_gradients(zip(grad, self.variables))
        self._display_action_probs()
        return len(self.rewards)

    def discount_rewards(self):
        self._display_rewards()
        return np.array(self.rewards), None

    def _display_action_probs(self):
        ap_after = [tf.nn.softmax(self.actor.get_policy(s))[0][a].numpy()
                    for a, s in zip(self.actions, self.states)]
        print('action probs: ', ['%.4f' % r for r in ap_after])

    def _display_actions(self):
        print('actions     : ', ['%.4f' % r for r in self.actions])

    def _display_rewards(self):
        print('rewards     : ', ['%.4f' % r for r in self.rewards])

    def _display_states(self):
        print('states      : ', ['%.4f' % r for r in
                                 self.states[:len(self.states)-1]])
