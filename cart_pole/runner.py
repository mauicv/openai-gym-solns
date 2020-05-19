from cart_pole.trainer import Trainer
from file_system_controller import FileWriter
import os
import tensorflow as tf


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data'):
        self.dirname = out
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        actor = None
        if os.path.isdir(self.model_loc):
            print('Loading model')
            actor = tf.keras.models.load_model(self.model_loc)
        else:
            print('No saved Model, Generating new one')
        self.trainer = Trainer(actor=actor)
        self.file_writer = FileWriter(self.dirname, 'scores')
        self.file_writer.init_file(['score'])

    @property
    def model_loc(self):
        return os.path.join(self.dirname, 'cart_pole')

    def start(self):
        for i in range(self.num_episodes):
            self.trainer.record_episode(self.num_steps)
            score = self.trainer.train()
            self.file_writer.write_val(score)
            self.trainer.reset()
            if i % 20 == 0:
                self.trainer.actor.model.save(self.model_loc)
            self.print(i, score)

    def print(self, i, score):
        print('-----------------------------')
        print('episode:', i)
        print('    score:', score)
        print('    length', self.trainer.episode_length)
