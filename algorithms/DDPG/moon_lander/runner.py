from algorithms.DDPG.moon_lander.trainer import Trainer
from file_system_controller import FileWriter
import os
import tensorflow as tf
import numpy as np
import gym
import math


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data', tau=0.01):
        self.dirname = out
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.scores = []

        critic = None
        if os.path.isdir(self.critic_loc):
            critic = tf.keras.models.load_model(self.critic_loc)

        actor = None
        if os.path.isdir(self.actor_loc):
            actor = tf.keras.models.load_model(self.actor_loc)

        self.trainer = Trainer(tau, actor=actor, critic=critic)
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def actor_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_ddpg', 'actor')

    @property
    def critic_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_ddpg', 'critic')

    @property
    def best_actor_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_ddpg', 'best_actor')

    @property
    def best_critic_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_ddpg', 'best_critic')

    def start(self):
        for i in range(self.num_episodes):
            score, learning_success_rate, episode_length = self.trainer\
                .run_episode()

            if i % 20 == 0:
                self.trainer.target_actor.model.save(self.actor_loc)
                self.trainer.target_critic.model.save(self.critic_loc)
            self.print(i, score, learning_success_rate)

            if i % 5 == 0:
                score = self.test_run()
                self.scores.append(score)
                score = np.array(self.scores[-25:]).mean()
                self.file_writer.write_val(score)
                if score >= max(self.scores):
                    self.trainer.target_actor.model.save(self.best_actor_loc)
                    self.trainer.target_critic.model.save(self.best_critic_loc)

    def print(self, i, score, learning_success_rate):
        print('----------------------------------')
        print('episode:', i)
        print('    length', self.trainer.episode_length)
        print('    score:', score)
        print('    learning sucess:', learning_success_rate)

    def test_run(self):
        done = False
        env = gym.make('LunarLanderContinuous-v2')
        state = env.reset()
        episode_length = 0
        episode_r = 0
        acc_Q_loss = 0

        while not done:
            episode_length += 1
            action = self.trainer.target_actor.model(state[np.newaxis, :])
            next_state, reward, done, _ = env.step(action[0])
            next_state = next_state[np.newaxis, :]
            next_action = self.trainer.target_actor.model(next_state)
            Q_input = tf.concat([next_state, next_action], axis=1)
            y = reward + self.trainer.discount_factor*self.trainer\
                .target_critic.model(Q_input)
            Q_input = tf.concat([state[np.newaxis, :], action], axis=1)
            pred_reward = (y - self.trainer.target_critic.model(Q_input))\
                .numpy()[0][0]

            acc_Q_loss += (reward - pred_reward)**2
            episode_r += reward
            state = next_state[0]

        print('-----------------------------------------------')
        print('Test run:')
        print('     ep len:', episode_length)
        print('     accumulated reward:', episode_r)
        print('     accumulated Q loss:', math.sqrt(acc_Q_loss))

        env.close()
        return episode_r
