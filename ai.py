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

def print_nboard(nboard):
    """ Pretty print board """
    print(nboard_to_str(nboard))

def check_winner_or_blocked(nboard):
    """ Check if a player won or their opponent is blocked """
    win = check_winner(nboard)
    if win == 0:
        if not allowed_moves(nboard, "b"):
            win = -1
        elif not allowed_moves(nboard, "w"):
            win = 1
    return win

def check_winner(nboard):
    """ Check if a play won """
    size = len(nboard)
    alive_b = False
    alive_w = False
    for i in range(size):
        for j in range(size):
            if nboard[i][j] > 0:
                alive_b = True
            elif nboard[i][j] < 0:
                alive_w = True
    if alive_b and not alive_w:
        return 1
    if not alive_b and alive_w:
        return -1
    return 0

def next_one_square(disc_type, ox, oy):
    """ return all one step """
    if disc_type == 1:
        return [(cx, cy) for (cx, cy) in [(ox+1, oy-1), (ox+1, oy+1)] if inboard(cx, cy)]
    elif disc_type == -1:
        return [(cx, cy) for (cx, cy) in [(ox-1, oy-1), (ox-1, oy+1)] if inboard(cx, cy)]
    else:
        return [(cx, cy) for (cx, cy) in [(ox-1, oy-1), (ox-1, oy+1), (ox+1, oy-1), (ox+1, oy+1)] if inboard(cx, cy)]

def depth_first_search(nboard, disc_type, ox, oy):
    """
    depth_first_search in graph
    """
    ret = {'capturing': [], 'non_capturing': [], 'capturing_list':[], 'end':True, 'type': disc_type}
    stack = [((ox, oy), ret)]
    while stack:
        ((ox, oy), ret) = stack.pop()
        if ret['capturing']:
            ret['end'] = True
            for (cx, cy) in next_one_square(ret['type'], ox, oy):
                if (cx, cy) not in ret['capturing_list'] and nboard[cx][cy] != 0 and np.sign(nboard[cx][cy]) != np.sign(ret['type']):
                    if inboard(2*cx-ox, 2*cy-oy) and (nboard[2*cx-ox][2*cy-oy] == 0 or (2*cx-ox, 2*cy-oy) == ret['capturing'][0]):
                        ret['end'] = False
                        stack.append(((2*cx-ox, 2*cy-oy), {'capturing_list': ret['capturing_list']+[(cx, cy)], 'non_capturing':ret['non_capturing'],\
                            'capturing': ret['capturing']+ [(2*cx-ox, 2*cy-oy)], 'end':False, 'type': get_disc_type(ret['type'], 2*cx-ox)}))

            if ret['end']:
                yield ret

        else:
            for (cx, cy) in next_one_square(ret['type'], ox, oy):
                if nboard[cx][cy] == 0:
                    yield {'capturing': [], 'non_capturing': [(ox, oy), (cx, cy)], 'capturing_list':[], 'end':True, 'type':get_disc_type(disc_type, cx, cy)}
                elif np.sign(nboard[cx][cy]) == np.sign(ret['type']):
                    yield {'capturing': [], 'non_capturing': [], 'capturing_list':[], 'end':True, 'type':disc_type}
                else:
                    if inboard(2*cx-ox, 2*cy-oy) and nboard[2*cx-ox][2*cy-oy] == 0:
                        stack.append(((2*cx-ox, 2*cy-oy), {'capturing_list':[(cx, cy)], 'capturing':[(ox, oy), (2*cx-ox, 2*cy-oy)], \
                            'non_capturing':[], 'end':False, 'type':get_disc_type(disc_type, 2*cx-ox)}))
                    else:
                        yield {'capturing': [], 'non_capturing': [], 'capturing_list':[], 'end':True, 'type':disc_type}
