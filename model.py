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

    def save_model(self):
        """ save model """
        filepath, ext = os.path.splitext(self.model_filepath)
        filepath = filepath + "_" + str(self.generation+1) # save for the next generation
        model_filepath = filepath + ext
        self.model.save(model_filepath)

    def load_model(self):
        """ load model """
        from keras.models import load_model
        filepath, ext = os.path.splitext(self.model_filepath)
        filepath = filepath + "_" + str(self.generation) # load for the generation
        model_filepath = filepath + ext
        self.model = load_model(model_filepath)

    def process_board(self, nboard):
        """ Separate board representation in 4 channels """
        repr_board = np.zeros((4, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if nboard[i, j] == 1:
                    repr_board[0, i, j] = 1
                elif nboard[i, j] == 2:
                    repr_board[1, i, j] = 1
                elif nboard[i, j] == -1:
                    repr_board[2, i, j] = 1
                elif nboard[i, j] == -2:
                    repr_board[3, i, j] = 1
        return repr_board.flatten()

    def learn(self, boards, values, epochs, batch_size, save=True):
        """ model learning """
        boards = [self.process_board(b) for b in boards]
        boards = np.stack(boards)
        values = np.array(values)
        self.model.fit(boards, values, epochs=epochs, batch_size=batch_size)
        if save:
            self.save_model()

    def predict(self, nboard):
        """ predict """
        repr_board = self.process_board(nboard)
        value = self.model.predict(repr_board.reshape(*(1,)+repr_board.shape)).squeeze()
        return value