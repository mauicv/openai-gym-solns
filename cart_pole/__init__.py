from cart_pole.runner import Runner
from gui import init_grapher
from multiprocessing import Process
import gym
import random
import tensorflow as tf


def play():
    env = gym.make('CartPole-v0')
    actor = Runner(0, 0).trainer.actor
    state = env.reset()
    for _ in range(1000):
        env.render()
        action = actor.get_action(state)
        policy = actor.get_policy(state)
        print('state:', state)
        print('action:', action)
        p = tf.nn.softmax(policy)[0, action].numpy()
        print('prob of action: ',  p)
        state, _, done, _ = env.step(action)
    env.close()


def train():
    p = Process(target=init_grapher,
                args=('./data', 'scores',))
    p.start()
    Runner(1000, 100).start()
    p.join()


def get_baseline():
    env = gym.make('CartPole-v0')
    state = env.reset()
    N = 1000
    counts = [0 for _ in range(N)]
    for i in range(N):
        done = False
        state = env.reset()
        while not done:
            counts[i] = counts[i] + 1
            state, _, done, _ = env.step(random.choice([0, 1]))
    env.close()
    print('average score:', sum(counts)/len(counts))
    return sum(counts)/len(counts)
