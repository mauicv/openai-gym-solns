import gym
from cart_pole.runner import Runner
from gui import init_grapher
from multiprocessing import Process


def play():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
    env.close()


def train():
    p = Process(target=init_grapher,
                args=('./data', 'scores',))
    p.start()
    Runner(100, 100).start()
    p.join()
