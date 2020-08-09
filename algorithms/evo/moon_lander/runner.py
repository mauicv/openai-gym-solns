from algorithms.evo.moon_lander.trainer import Trainer
from file_system_controller import FileWriter
import os
import tensorflow as tf
import numpy as np
import gym
# import math


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data', tau=0.01):
        self.dirname = out
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.scores = []

        actor = None
        if os.path.isdir(self.actor_loc):
            actor = tf.keras.models.load_model(self.actor_loc)

        self.trainer = Trainer(actor=actor)
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def actor_loc(self):
        return os.path.join(self.dirname, 'lunar_lander_evo', 'actor')

    def start(self):
        for i in range(self.num_episodes):
            score = self.trainer.run_episode()
            self.scores.append(score)
            score = np.array(self.scores[-25:]).mean()
            self.file_writer.write_val(score)

            if i % 20 == 0:
                self.trainer.actor.model.save(self.actor_loc)

            self.print(i, score)

            if i % 5 == 0:
                score = self.test_run()

    def print(self, i, score):
        print('----------------------------------')
        print('episode:', i)
        print('    length', self.trainer.episode_length)
        print('    score:', score)

    def test_run(self):
        done = False
        env = gym.make('LunarLanderContinuous-v2')
        state = env.reset()
        episode_length = 0
        episode_r = 0

        while not done:
            episode_length += 1
            action = self.trainer.actor.model(state[None, :])
            next_state, reward, done, _ = env.step(action[0])
            episode_r += reward
            state = next_state

        print('-----------------------------------------------')
        print('Test run:')
        print('     accumulated reward:', episode_r)

        env.close()
        return episode_r
