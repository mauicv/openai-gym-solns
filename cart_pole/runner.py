from cart_pole.trainer import Trainer
from file_system_controller import FileWriter


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data'):
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.trainer = Trainer()
        self.file_writer = FileWriter('./data', 'scores')
        self.file_writer.init_file(['score'])

    def start(self):
        for i in range(self.num_episodes):
            self.trainer.record_episode(self.num_steps)
            score = self.trainer.train()
            self.file_writer.write_val(score)
            self.trainer.reset()
            self.print(i, score)

    def print(self, i, score):
        print('-----------------------------')
        print('episode:', i)
        print('    score:', score)
        print('    length', self.trainer.episode_length)
