"""
    ai
"""
import random
import numpy as np
from model import ValueNet

def inboard(x, y, size=8):
    """ check if (x,y) in board or not """
    return (0 <= x < size) and (0 <= y < size)

def get_disc_type(origin_type, x, size=8):
    """ check disc type to king or not """
    disc_type = origin_type
    if origin_type == -1 and x == 0:
        disc_type = -2
    elif origin_type == 1 and x == size-1:
        disc_type = 2
    return disc_type

def board_to_numpy(board):
    """ Convert a list of string nboard to a numpy 2d array """
    if isinstance(board, np.ndarray):
        return board
    size = len(board)
    nboard = np.zeros((size, size), dtype=np.int64)
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            if square == 'b':
                nboard[i, j] = 1
            elif square == 'B':
                nboard[i, j] = 2
            elif square == 'w':
                nboard[i, j] = -1
            elif square == 'W':
                nboard[i, j] = -2
    return nboard

def new_board():
    board = [
        '_b_b_b_b',
        'b_b_b_b_',
        '_b_b_b_b',
        '________',
        '________',
        'w_w_w_w_',
        '_w_w_w_w',
        'w_w_w_w_'
        ]
    return board_to_numpy(board)

def nboard_to_str(nboard):
    """ Pretty board repr """
    size = nboard.shape[0]
    num_to_letter = {1: "b", 2: "B", -1: "w", -2: "W"}
    sep = "   " + "+---" * size + "+"
    output_string = "     " + "   ".join(str(i) for i in range(size))
    output_string += "\n" + sep
    for i, row in enumerate(nboard):
        line = " " + str(i) + " | " + " | ".join(num_to_letter.get(num, " ") for num in row) + " |"
        output_string += "\n" + line + "\n" + sep
    return output_string
