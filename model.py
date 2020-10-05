"""
    neural network model
"""
import os
import numpy as np

class ValueNet():
    """policy-value network """
    def __init__(self, model_filepath, generation, size=8, l2_penalty=1e-4):
        self.size = size
        self.l2_penalty = l2_penalty
        self.model_filepath = model_filepath
        self.generation = generation
        if generation >= 0:
            self.load_model()
        else:
            self.create_value_network()

    def create_value_network(self):
        """create the policy value network """
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras import regularizers
        self.model = Sequential()
        self.model.add(Dense(64,
                             input_shape=(4 * self.size * self.size,),
                             use_bias=True,
                             kernel_initializer='random_uniform',
                             kernel_regularizer=regularizers.l2(self.l2_penalty),
                             bias_initializer='zeros',
                             activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(16,
                             use_bias=True,
                             kernel_initializer='random_uniform',
                             kernel_regularizer=regularizers.l2(self.l2_penalty),
                             bias_initializer='zeros',
                             activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1,
                             use_bias=True,
                             kernel_initializer='random_uniform',
                             kernel_regularizer=regularizers.l2(self.l2_penalty),
                             bias_initializer='zeros',
                             activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
