import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import nn
import numpy as np


tf.keras.backend.set_floatx('float64')


class Actor:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def init_model(cls, num_layers, input_dim, layer_dim, output_dim):
        model = Sequential()
        model.add(Dense(layer_dim, input_dim=input_dim, activation='relu'))
        for layer in range(num_layers):
            model.add(Dense(
                layer_dim, activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        model.add(Dense(output_dim))
        return cls(model=model)

    def get_policy(self, inputs):
        return self.model(inputs[np.newaxis, :])

    def get_action(self, inputs):
        policy = nn.softmax(self.get_policy(inputs))
        action = policy.numpy().argmax()
        return action


class ContinuousActor:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def init_model(cls, num_layers, input_dim, layer_dim, output_dim):
        model = Sequential()
        model.add(Dense(layer_dim, input_dim=input_dim, activation='relu'))
        for layer in range(num_layers):
            model.add(Dense(
                layer_dim, activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        model.add(Dense(output_dim,  activation='tanh',))
        return cls(model=model)

    def get_action(self, inputs):
        return self.model(inputs[np.newaxis, :])[0]

    def track_weights(self, tau, model):
        self_weights = self.model.get_weights()
        other_weights = model.get_weights()
        other_weights = [(1 - tau) * self_weight + tau *
                         other_weight for self_weight, other_weight in
                         zip(self_weights, other_weights)]
        self.model.set_weights(other_weights)


class Critic:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def init_model(cls, num_layers, input_dim, layer_dim):
        model = Sequential()
        model.add(Dense(layer_dim, input_dim=input_dim, activation='relu'))
        for layer in range(num_layers):
            model.add(Dense(
                layer_dim, activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        model.add(Dense(1))
        return cls(model=model)

    def get_Q(self, state, one_hot_action):
        inputs = np.concatenate([state, one_hot_action])
        return self.model(inputs[np.newaxis, :])

    def track_weights(self, tau, model):
        self_weights = self.model.get_weights()
        other_weights = model.get_weights()
        other_weights = [(1 - tau) * self_weight + tau *
                         other_weight for self_weight, other_weight in
                         zip(self_weights, other_weights)]
        self.model.set_weights(other_weights)
