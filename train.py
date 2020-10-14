"""
    main train
"""
import os
import random
from collections import deque, Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from ai import play_game, RLValueMinimaxPlayer, SimpleMinimaxPlayer
from model import ValueNet

# PARAMS
board_size = 8
base_minimax_depth = 6
rl_minimax_depth = 6
random_burn_in = 2
trace_min = 2
limit_to_draw = 200
num_cores = 8
# board generation params
buffer_size = 20000
num_init_games = 16
num_rl_games = 8
num_test_games = 8
# training params
rl_rounds = 100
batch_size = 256
learn_rate = 2e-3
epochs = 30
test_freq = 5
rl_model_filepath = "./mlp_200_model.h5"  #multi-layer perceptron

