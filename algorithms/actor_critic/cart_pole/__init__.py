from algorithms.actor_critic.cart_pole.runner import Runner
from gui import init_grapher
from multiprocessing import Process
import gym
# import numpy as np


def play(eps, steps):
    env = gym.make('CartPole-v0')
    trainer = Runner(0, 0).trainer
    actor = trainer.actor

    state = env.reset()
    for _ in range(steps):
        env.render()
        action = actor.get_action(state)
        action_Q = trainer.get_action_q_value(state, action)
        print(action_Q.numpy()[0][0])
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
    trainer = Runner(0, 0, out='./algorithms/actor_critic/cart_pole/soln') \
        .trainer
    actor = trainer.actor
    state = env.reset()
    for _ in range(steps):
        env.render()
        action = actor.get_action(state)
        # action_Q = trainer.get_action_q_value(state, action)
        # print(action_Q.numpy()[0][0])
        state, _, done, _ = env.step(action)
    env.close()
