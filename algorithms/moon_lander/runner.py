from algorithms.moon_lander.trainer import Trainer
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
        actor = None
        if os.path.isdir(self.model_loc):
            actor = tf.keras.models.load_model(self.model_loc)
        self.trainer = Trainer(actor=actor)
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def model_loc(self):
        return os.path.join(self.dirname, 'lunder_lander')

    def start(self):
        for i in range(self.num_episodes):
            self.trainer.record_episode(self.num_steps)
            score = self.trainer.train()
            self.scores.append(score)
            print(score)
            score = np.array(self.scores[-25:]).mean()
            self.file_writer.write_val(score)
            self.print(i, score)
            self.trainer.reset()
            if i % 20 == 0:
                self.trainer.actor.model.save(self.model_loc)

    def print(self, i, score):
        print('----------------------------------')
        print('episode:', i)
        print('    length', len(self.trainer.states))
        print('    score:', score)
