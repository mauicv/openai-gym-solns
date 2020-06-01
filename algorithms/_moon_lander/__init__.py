from algorithms._moon_lander.runner import Runner
import gym
from gui import init_grapher
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def play_moon_lander_path():
    env = gym.make('LunarLander-v2')
    trainer = Runner(0, 0).trainer
    actor = trainer.actor
    state = env.reset()
    rewards = []
    done = False
    states = [state]
    actions = []
    while not done:
        env.render()
        policy = actor.get_policy(state)
        action = trainer.sample_action(policy)
        state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    env.close()
    # print(sum(rewards))


def train_moon_lander_path(eps, steps=None):
    p_1 = Process(target=init_grapher,
                  args=('./data', 'scores', 'score', 'left'))
    p_2 = Process(target=init_grapher,
                  args=('./data', 'path-stat', 'path_seperation_avg',
                        'middle'))
    p_3 = Process(target=init_grapher,
                  args=('./data', 'path-stat', 'path_seperation_var', 'right'))
    p_1.start()
    p_2.start()
    p_3.start()
    Runner(eps, steps).start()
    p_1.join()
    p_2.join()
    p_3.join()


def compute_path_stat_moon_lander():
    env = gym.make('LunarLander-v2')
    trainer = Runner(0, 0).trainer
    actor = trainer.actor

    N = 20
    states = [[] for _ in range(N)]
    actions = [[] for _ in range(N)]
    rewards = [[] for _ in range(N)]
    for i in tqdm(range(N)):
        state = env.reset()
        states[i].append(state)
        done = False
        while not done:
            policy = actor.get_policy(state)
            action = trainer.sample_action(policy)
            # action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            states[i].append(state)
            actions[i].append(action)
            rewards[i].append(reward)
    env.close()

    mean_dist, std_dist = avg_dists(states)

    fig, ax = plt.subplots()
    ax.plot(mean_dist, label='Separation mean')
    ax.plot(std_dist, label='Separation standard deviation')
    ax.legend(frameon=False, loc='lower center', ncol=2)
    ax.legend()
    plt.show()


def avg_dists(states):
    max_len = max([len(item) for item in states])
    for i, states_set in enumerate(states):
        padding = ((0, max_len - len(states_set)), (0, 0))
        states[i] = np.pad(states_set, padding, mode='edge')

    states = np.array(states)

    mean_diff_vects = states - np.mean(states, axis=2)[:, :, None]
    mean_dist = np.mean(np.sqrt(np.sum((mean_diff_vects)**2, axis=2)), axis=0)
    std_dist = np.std(np.sqrt(np.sum((mean_diff_vects)**2, axis=2)), axis=0)
    return mean_dist, std_dist
