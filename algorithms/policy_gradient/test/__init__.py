from algorithms.policy_gradient.test.runner import Runner
import numpy as np


def play_test(runner):
    env = runner.trainer.env
    actor = runner.trainer.actor
    state = env.reset()
    reward_sum = 0
    for _ in range(10):
        action = actor.get_action(state)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
    return reward_sum


def train_test(eps=1000, steps=10):
    runner = Runner(eps, steps)

    rewards = np.zeros(eps)
    for i in range(eps):
        rewards[i] = play_test(runner)
    print('Mean reward before training: ', rewards.mean())

    runner.start()

    rewards = np.zeros(eps)
    for i in range(steps):
        rewards[i] = play_test(runner)
    print('Mean reward after training: ', rewards.mean())
