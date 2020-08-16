from algorithms.TD3.gt_stander.trainer import Trainer
from file_system_controller import FileWriter
import os
import tensorflow as tf
import numpy as np
import pygtgym as gym
import math


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data', tau=0.01):
        self.dirname = out
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.scores = []

        critic_1 = None
        if os.path.isdir(self.critic_1_loc):
            critic_1 = tf.keras.models.load_model(self.critic_1_loc)

        critic_2 = None
        if os.path.isdir(self.critic_2_loc):
            critic_2 = tf.keras.models.load_model(self.critic_2_loc)

        actor = None
        if os.path.isdir(self.actor_loc):
            actor = tf.keras.models.load_model(self.actor_loc)

        self.trainer = Trainer(tau, actor=actor, critics=[critic_1, critic_2])
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def actor_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'actor')

    @property
    def critic_1_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'critic_1')

    @property
    def critic_2_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'critic_2')

    @property
    def best_actor_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'best_actor')

    @property
    def best_critic_1_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'best_critic_1')

    @property
    def best_critic_2_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_td3', 'best_critic_2')

    def start(self):
        for i in range(self.num_episodes):
            reward_sum, episode_length = \
                self.trainer.run_episode()

            if i % 20 == 0:
                self.trainer.actor.model.save(self.actor_loc)
                self.trainer.critic_1.model.save(self.critic_1_loc)
                self.trainer.critic_2.model.save(self.critic_2_loc)

            self.print(i, reward_sum, episode_length)

            if i % 5 == 0:
                score = self.test_run()
                self.scores.append(score)
                score = np.array(self.scores[-25:]).mean()
                self.file_writer.write_val(score)
                if score >= max(self.scores):
                    self.trainer.actor.model.save(self.best_actor_loc)
                    self.trainer.critic_1.model.save(self.best_critic_1_loc)
                    self.trainer.critic_2.model.save(self.best_critic_2_loc)

    def print(self, i, reward_sum, episode_length):
        print('----------------------------------')
        print('episode:', i)
        print('    length', episode_length)
        print('    reward_sum:', reward_sum)

    def test_run(self):
        done = False
        env = gym.make('LunarLanderContinuous-v2')
        state = env.reset()
        episode_length = 0
        episode_r = 0
        acc_Q_loss_1 = 0
        acc_Q_loss_2 = 0

        while not done:
            episode_length += 1
            action = self.trainer.actor.model(state[np.newaxis, :])
            next_state, reward, done, _ = env.step(action[0])
            next_state = next_state[np.newaxis, :]
            next_action = self.trainer.actor.model(next_state)

            Q_input = tf.concat([next_state, next_action], axis=1)
            Q_1_val = self.trainer.target_critic_1.model(Q_input)
            Q_2_val = self.trainer.target_critic_1.model(Q_input)
            Q_val = tf.math.minimum(Q_1_val, Q_2_val)
            y = reward + self.trainer.discount_factor*Q_val
            Q_input = tf.concat([state[np.newaxis, :], action], axis=1)
            Q_1 = self.trainer.critic_1.model(Q_input)
            Q_2 = self.trainer.critic_2.model(Q_input)
            pred_reward_1 = (y - Q_1).numpy()[0][0]
            pred_reward_2 = (y - Q_2).numpy()[0][0]
            acc_Q_loss_1 += (reward - pred_reward_1)**2
            acc_Q_loss_2 += (reward - pred_reward_2)**2
            episode_r += reward
            state = next_state[0]

        print('*****************Evaluation run:**********************')
        print('     ep len:', episode_length)
        print('     accumulated reward:', episode_r)
        print('     accumulated Q_1 loss:', math.sqrt(acc_Q_loss_1))
        print('     accumulated Q_2 loss:', math.sqrt(acc_Q_loss_2))
        print('******************************************************')

        env.close()
        return episode_r
