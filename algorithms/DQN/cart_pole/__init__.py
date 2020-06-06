from algorithms.DQN.cart_pole.runner import Runner
from gui import init_grapher
from multiprocessing import Process
import gym
import numpy as np


def play(eps, steps):
    env = gym.make('CartPole-v0')
    trainer = Runner(0, 0).trainer

    state = env.reset()
    for _ in range(steps):
        env.render()
        action_Qs = trainer.get_action_vals(state)
        action = np.argmax(action_Qs)
        state, _, done, _ = env.step(action)
    env.close()


def train(eps, steps):
    p = Process(target=init_grapher,
                args=('./data', 'scores',))
    p.start()
    Runner(eps, steps).start()
    p.join()


def example(eps, steps):
    env = gym.make('CartPole-v0')
    trainer = Runner(0, 0, out='./algorithms/DQN/cart_pole/soln') \
        .trainer

    state = env.reset()
    for _ in range(steps):
        env.render()
        action_Qs = trainer.get_action_vals(state)
        action = np.argmax(action_Qs)
        state, _, done, _ = env.step(action)
    env.close()
