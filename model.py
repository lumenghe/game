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
