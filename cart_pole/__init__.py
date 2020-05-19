import gym
from cart_pole.trainer import Trainer
from gui.graphing import grapher_and_train
import asyncio
from threading import Thread, Event
from time import sleep
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(threadName)10s %(name)18s: %(message)s',
    stream=sys.stderr,
)


def play():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
    env.close()


def train():
    Runner(100, 100).start()


def train_async():
    kill_sig = Runner(100, 100).async_start()
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            kill_sig.set()
            break


def train_and_graph():
    train = Runner(100, 100)
    grapher_and_train(train)


class Runner:
    def __init__(self, num_episodes, num_steps):
        self.xs = []
        self.ys = []
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.trainer = Trainer()
        self.threads = []
        self.loop = None
        self.event = None

    def start(self):
        for i in range(self.num_episodes):
            if self.event.is_set():
                self.loop.stop()
            self.trainer.record_episode(self.num_steps)
            score = self.trainer.train()
            self.xs.append(i)
            self.ys.append(score)
            self.trainer.reset()
            self.print_outcomes()
        self.loop.stop()

    def print_outcomes(self):
        print('-----------------------------')
        print('episode:', self.xs[len(self.xs)-1])
        print('    score:', self.ys[len(self.ys)-1])
        print('    length', self.trainer.episode_length)

    def async_start(self):
        self._start_process(self.start)
        self.event = Event()
        return self.event

    def _start_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _start_process(self, process):
        self.loop = asyncio.new_event_loop()
        t = Thread(target=self._start_loop, args=(self.loop,))
        t.start()
        self.loop.call_soon_threadsafe(process)
        self.threads.append(t)
