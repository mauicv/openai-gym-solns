from cart_pole.runner import Runner
from gui import init_grapher
from multiprocessing import Process
import gym


def play():
    env = gym.make('CartPole-v0')
    actor = Runner(0, 0).trainer.actor
    state = env.reset()
    for _ in range(100):
        env.render()
        action = actor.get_action(state)
        env.step(int(action))
    env.close()


def train():
    p = Process(target=init_grapher,
                args=('./data', 'scores',))
    p.start()
    Runner(100, 100).start()
    p.join()
