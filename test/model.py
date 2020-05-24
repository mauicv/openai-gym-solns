import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import nn


tf.keras.backend.set_floatx('float64')


class Actor:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def init_model(cls, num_layers, input_dim, layer_dim, output_dim):
        model = Sequential()
        model.add(Dense(layer_dim, input_dim=input_dim, activation='relu'))
        for layer in range(num_layers):
            model.add(Dense(layer_dim, activation='relu'))
            model.add(Dense(output_dim))
        return cls(model=model)

    def get_policy(self, inputs):
        return self.model(inputs[None, :])

    def get_action(self, inputs):
        # print(inputs)
        policy = nn.softmax(self.get_policy(inputs))
        action = policy.numpy().argmax()
        return action
