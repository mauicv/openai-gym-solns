from moon_lander.runner import Runner
import gym
from gui import init_grapher
from multiprocessing import Process
import random


def play():
    env = gym.make('LunarLander-v2')
    trainer = Runner(0, 0).trainer
    actor = trainer.actor
    state = env.reset()
    rewards = []
    done = False
    while not done:
        env.render()
        # action = actor.get_action(state)
        policy = actor.get_policy(state)
        action = trainer.sample_action(policy)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
    env.close()
    print(sum(rewards))


def train():
    p = Process(target=init_grapher,
                args=('./data', 'scores',))
    p.start()
    Runner(1000, None).start()
    p.join()


def get_baseline():
    env = gym.make('LunarLander-v2')
    state = env.reset()
    N = 1000
    counts = [0 for _ in range(N)]
    for i in range(N):
        done = False
        state = env.reset()
        while not done:
            counts[i] = counts[i] + 1
            state, _, done, _ = env.step(random.choice([0, 1, 2, 3]))
    env.close()
    print('average score:', sum(counts)/len(counts))
    return sum(counts)/len(counts)


# def play_trained_soln():
#     env = gym.make('LunarLander-v2')
#     actor = Runner(0, 0, out='./lunder_lander/soln').trainer.actor
#     state = env.reset()
#     for _ in range(1000):
#         env.render()
#         action = actor.get_action(state)
#         state, _, done, _ = env.step(action)
#     env.close()
