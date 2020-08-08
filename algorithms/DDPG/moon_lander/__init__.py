from algorithms.DDPG.moon_lander.runner import Runner
from multiprocessing import Process
import gym
from gui import init_grapher
# import tensorflow as tf

# import numpy as np


def play(eps, steps):
    env = gym.make('LunarLanderContinuous-v2')
    trainer = Runner(0, 0).trainer
    actor = trainer.actor
    # print(env.action_space.high[0])
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = actor.get_action(state)
        # action = action + tf.random\
        #     .normal([2], mean=0.0, stddev=1.0,
        #             dtype=tf.dtypes.float64)
        # qs = trainer.get_Q_value(state, action)
        # print(qs.numpy())
        # print(action)
        # action_Q = trainer.get_Q_value(state, action)
        # print(action_Q.numpy()[0][0])
        state, _, done, _ = env.step(action)
    env.close()


def train(eps, steps):
    if init_grapher:
        p = Process(target=init_grapher,
                    args=('./data', 'scores',))
        p.start()
        Runner(eps, steps, tau=0.01).start()
        p.join()
    else:
        Runner(eps, steps, tau=0.01).start()


def test_Q(eps=1, steps=200):
    r = Runner(eps, steps, tau=0.01)
    r.test_run()


def example(eps, steps):
    env = gym.make('LunarLanderContinuous-v2')
    trainer = Runner(0, 0, out='./algorithms/DDPG/moon_lander/soln') \
        .trainer
    actor = trainer.actor
    state = env.reset()
    done = False
    i = 0
    while not done:
        i = i + 1
        env.render()
        action = actor.get_action(state)
        # action_Q = trainer.get_Q_value(state, action)
        # print(action_Q.numpy()[0][0])
        state, _, done, _ = env.step(action)
        done = i > 5000
    env.close()
