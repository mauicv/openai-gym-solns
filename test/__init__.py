from test.runner import Runner
import numpy as np


def play(runner):
    env = runner.trainer.env
    actor = runner.trainer.actor
    state = env.reset()
    reward_sum = 0
    for _ in range(10):
        action = actor.get_action(state)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
    return reward_sum


def train():
    TRAINING_SET_SIZE = 100

    runner = Runner(TRAINING_SET_SIZE, 10)

    rewards = np.zeros(TRAINING_SET_SIZE)
    for i in range(TRAINING_SET_SIZE):
        rewards[i] = play(runner)
    print('Mean reward before training: ', rewards.mean())

    runner.start()

    rewards = np.zeros(TRAINING_SET_SIZE)
    for i in range(TRAINING_SET_SIZE):
        rewards[i] = play(runner)
    print('Mean reward after training: ', rewards.mean())
