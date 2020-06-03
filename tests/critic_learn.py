from algorithms.model import Critic
import tensorflow as tf
import numpy as np
from random import shuffle, random


def test_critic():
    critic = Critic.init_model(2, 4, 100)
    one_hot_action = tf.one_hot([1], 2)
    Q = critic.get_V(np.array([0, 0]), one_hot_action)
    print(Q)


def test_critic_A():
    length = 200000
    val = 0.3

    critic = Critic.init_model(2, 4, 100)
    critic.model.compile(optimizer='adam',
                         loss=tf.keras.losses.MeanSquaredError())

    x_train = np.array([[0.7, 0.1, 0, 0.5]]*length)
    y_train = np.random.normal(val, 1, length)

    critic.model.fit(x_train, y_train, epochs=5)

    r = critic.get_q(np.array([0.7, 0.1, 0, 0.5]))
    print('estimate:', r.numpy()[0][0])
    assert abs(r - val) < 0.01


def test_critic_B():
    length = 100000
    val_1 = 0.3
    val_2 = 0.5

    critic = Critic.init_model(2, 2, 100)
    critic.model.compile(optimizer='adam',
                         loss=tf.keras.losses.MeanSquaredError())

    x_train = []
    y_train = []
    for i in range(length):
        if random() > 0.5:
            x_train.append([1, 0])
            y_train.append(np.random.normal(val_1, 0.5, 1)[0])
        else:
            x_train.append([0, 1])
            y_train.append(np.random.normal(val_2, 0.5, 1)[0])

    all = list(zip(x_train, y_train))
    shuffle(all)

    x_train = [a for a, _ in all]
    y_train = [b for _, b in all]

    critic.model.fit(x_train, y_train, epochs=5)

    r = critic.get_q(np.array([1, 0]))
    print('estimate:', r.numpy()[0][0],
          'accuracy: ', abs(r - val_1).numpy()[0][0])
    r = critic.get_q(np.array([0, 1]))
    print('estimate:', r.numpy()[0][0],
          'accuracy: ', abs(r - val_2).numpy()[0][0])
