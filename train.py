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


if __name__ == '__main__':
    # store boards and labels
    data_buffer = deque(maxlen=buffer_size)
    # initialize with boards from minimax
    print("\nInitializing data with base minimax strategy ({} GAMES)\n".format(num_init_games))
    game_ids = range(num_init_games)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        def run_init_game(i):
            print("Generating game number {}".format(i))
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            minimax_player = SimpleMinimaxPlayer(base_minimax_depth)
            _, _, trace = play_game(minimax_player, minimax_player, verbose=1, limit_to_draw=limit_to_draw, random_burn_in=random_burn_in, trace_min=trace_min)
            return trace
        for trace in executor.map(run_init_game, game_ids):
            data_buffer.extend(trace)

    # print main learning loop
    for rl_step in range(0, rl_rounds):
        # add examples from previous model (each process needs to load last generation model)
        if rl_step > 0:
            print("\nAdding data from previous generation model ({} GAMES)\n".format(num_rl_games))
            game_ids = range(num_rl_games)
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                def run_rl_game(i):
                    print("Generating RL game number {} from generation {}".format(i, rl_step-1))
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    value_net = ValueNet(rl_model_filepath, rl_step-1) # load from previous generation
                    rl_player = RLValueMinimaxPlayer(value_net, rl_minimax_depth)
                    _, _, trace = play_game(rl_player, rl_player, verbose=1, limit_to_draw=limit_to_draw, random_burn_in=random_burn_in, trace_min=trace_min)
                    return trace
                for trace in executor.map(run_rl_game, game_ids):
                    data_buffer.extend(trace)
        print("\nRL step {}, learning using {} examples\n".format(rl_step, len(data_buffer)))
        # learn new generation model
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            def learn(i):
                boards, values = zip(*data_buffer)
                value_net = ValueNet(rl_model_filepath, rl_step-1)
                value_net.learn(boards, values, epochs=epochs, batch_size=batch_size)
            executor.map(learn, [0])
        # test vs base model
        if rl_step % test_freq == 0:
            print("\nTesting vs simple minimax")
            game_ids = list(range(num_test_games))
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                def run_test_game(i):
                    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                    value_net = ValueNet(rl_model_filepath, rl_step-1) # load from previous generation
                    rl_player = RLValueMinimaxPlayer(value_net, rl_minimax_depth)
                    minimax_player = SimpleMinimaxPlayer(base_minimax_depth)
                    if i % 2:
                        win, step, _ = play_game(rl_player, minimax_player, verbose=0, limit_to_draw=limit_to_draw)
                    else:
                        win, step, _ = play_game(minimax_player, rl_player, verbose=0, limit_to_draw=limit_to_draw)
                    return win, step
                outcome = {"RL Blacks": Counter(), "RL Whites": Counter()}
                steps = {"RL Blacks": {}, "RL Whites": {}}
                for i, (win, step) in zip(game_ids, executor.map(run_test_game, game_ids)):
                    if i % 2:
                        outcome["RL Blacks"][win] += 1
                        steps["RL Blacks"].setdefault(win, []).append(str(step))
                    else:
                        outcome["RL Whites"][win] += 1
                        steps["RL Whites"].setdefault(win, []).append(str(step))
                print("\nRL Blacks win count: {}".format(outcome["RL Blacks"][1]))
                print("RL Blacks lose count: {}".format(outcome["RL Blacks"][-1]))
                print("RL Blacks draw count: {}".format(outcome["RL Blacks"][0]))
                print("\nRL Blacks win steps: {}".format(", ".join(steps["RL Blacks"].get(1, []))))
                print("RL Blacks lose steps: {}".format(", ".join(steps["RL Blacks"].get(-1, []))))
                print("RL Blacks draw steps: {}".format(", ".join(steps["RL Blacks"].get(0, []))))
                print("\nRL Whites win count: {}".format(outcome["RL Whites"][-1]))
                print("RL Whites lose count: {}".format(outcome["RL Whites"][1]))
                print("RL Whites draw count: {}".format(outcome["RL Whites"][0]))
                print("\nRL Whites win steps: {}".format(", ".join(steps["RL Whites"].get(-1, []))))
                print("RL Whites lose steps: {}".format(", ".join(steps["RL Whites"].get(1, []))))
                print("RL Whites draw steps: {}".format(", ".join(steps["RL Whites"].get(0, []))))