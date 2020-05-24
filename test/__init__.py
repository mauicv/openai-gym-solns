from test.runner import Runner
# from gui import init_grapher
# from multiprocessing import Process
# import random
# import tensorflow as tf
import numpy as np
# from test.env import Env


def play(runner):
    env = runner.trainer.env
    actor = runner.trainer.actor
    state = env.reset()
    reward_sum = 0
    for _ in range(10):
        action = actor.get_action(state)
        # policy = actor.get_policy(state)
        print('*******************************************')
        print('state:', state)
        print('action:', action)
        print('success: ', action != state[0])
        # p = tf.nn.softmax(policy)[0, action].numpy()
        # print('prob of action: ',  p)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        # print('reward: ', reward)
        # print('*******************************************')
    return reward_sum


def train():
    # p = Process(target=init_grapher,
    #             args=('./data', 'scores',))
    # p.start()
    TRAINING_SET_SIZE = 10

    runner = Runner(TRAINING_SET_SIZE, 10)

    rewards = np.zeros(TRAINING_SET_SIZE)
    for i in range(TRAINING_SET_SIZE):
        rewards[i] = play(runner)
    print(rewards)
    print(rewards.mean())

    runner.start()

    rewards = np.zeros(TRAINING_SET_SIZE)
    for i in range(TRAINING_SET_SIZE):
        rewards[i] = play(runner)
    print(rewards)
    print(rewards.mean())
    # p.join()


# def get_baseline():
#     env = gym.make('CartPole-v0')
#     state = env.reset()
#     N = 1000
#     counts = [0 for _ in range(N)]
#     for i in range(N):
#         done = False
#         state = env.reset()
#         while not done:
#             counts[i] = counts[i] + 1
#             state, _, done, _ = env.step(random.choice([0, 1]))
#     env.close()
#     print('average score:', sum(counts)/len(counts))
#     return sum(counts)/len(counts)
