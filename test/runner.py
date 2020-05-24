from test.trainer import Trainer


class Runner:
    def __init__(self, num_episodes, num_steps, out='./data'):
        self.dirname = out
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.trainer = Trainer()

    def start(self):
        for i in range(self.num_episodes):
            self.trainer.record_episode(self.num_steps)
            score = self.trainer.train()
            self.trainer.reset()
            self.print(i, score)

    def print(self, i, score):
        pass
        # print('----------------------------------')
        # print('episode:', i)
        # print('    score:', score)
        # print('    length', self.trainer.episode_length)
