from algorithms.actor_critic.cart_pole.trainer import Trainer
from file_system_controller import FileWriter
import os
import tensorflow as tf
import numpy as np


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data'):
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

        self.trainer = Trainer(actor=actor, critic=critic)
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def actor_loc(self):
        return os.path.join(self.dirname, 'cart_pole_ac', 'actor')

    @property
    def critic_loc(self):
        return os.path.join(self.dirname, 'cart_pole_ac', 'critic')

    def start(self):
        for i in range(self.num_episodes):
            score = self.trainer.record_episode(self.num_steps)
            self.scores.append(score)
            score = np.array(self.scores[-25:]).mean()
            self.file_writer.write_val(score)
            if i % 20 == 0:
                self.trainer.actor.model.save(self.actor_loc)
                self.trainer.critic.model.save(self.critic_loc)
            self.print(i, score)

    def print(self, i, score):
        print('----------------------------------')
        print('episode:', i)
        print('    length', self.trainer.episode_length)
        print('    score:', score)
