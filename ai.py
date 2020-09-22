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

def _apply_non_capture(nboard, i, j, next_i, next_j):
    """ Apply a non-capturing step """
    size = len(nboard)
    nboard[next_i][next_j] = nboard[i][j]
    if nboard[i][j] > 0 and next_i == size-1:
        nboard[next_i][next_j] = 2
    elif nboard[i][j] < 0 and next_i == 0:
        nboard[next_i][next_j] = -2 # Promotion to King if needed
    nboard[i][j] = 0

def _apply_capture(nboard, i, j, next_i, next_j):
    """ Apply a capturing step """
    size = len(nboard)
    nboard[next_i][next_j] = nboard[i][j]
    if nboard[i][j] > 0 and next_i == size-1:
        nboard[next_i][next_j] = 2
    elif nboard[i][j] < 0 and next_i == 0:
        nboard[next_i][next_j] = -2 # Promotion to King if needed
    nboard[i][j] = 0
    nboard[(i + next_i) // 2][(j + next_j) // 2] = 0

def apply_move(nboard, move):
    """ Modify the nboard when playing a move """
    retboard = nboard.copy()
    for (i, j), (next_i, next_j) in zip(move[:-1], move[1:]):
        if abs(i - next_i) == 1: # sufficient to say it is a non capturing step
            _apply_non_capture(retboard, i, j, next_i, next_j)
        else:
            _apply_capture(retboard, i, j, next_i, next_j)
    return retboard

def simple_with_end_eval(nboard):
    """ +1 for black disk +2 for black king
        -1 for white disk -2 for white king
        + 24 for black winning
        - 24 for white winning """
    cwob = check_winner_or_blocked(nboard)
    if cwob:
        return cwob
    return np.sum(nboard) / 24.0

def allowed_moves(board, color):
    """
        This is the first function you need to implement.

        Arguments:
        - board: The content of the board, represented as a list of strings.
                 The length of strings are the same as the length of the list,
                 which represents a NxN checkers board.
                 Each string is a row, from the top row (the black side) to the
                 bottom row (white side). The string are made of five possible
                 characters:
                 - '_' : an empty square
                 - 'b' : a square with a black disc
                 - 'B' : a square with a black king
                 - 'w' : a square with a white disc
                 - 'W' : a square with a white king
                 At the beginning of the game:
                 - the top left square of a board is always empty
                 - the square on it right always contains a black disc
        - color: the next player's color. It can be either 'b' for black or 'w'
                 for white.

        Return value:
        It must return a list of all the valid moves. Please refer to the
        README for a description of what are valid moves. A move is a list of
        all the squares visited by a disc or a king, from its initial position
        to its final position. The coordinates of the square must be specified
        using (row, column), with both 'row' and 'column' starting from 0 at
        the top left corner of the board (black side).

        Example:
        >> board = [
            '________',
            '__b_____',
            '_w_w____',
            '________',
            '_w______',
            '_____b__',
            '____w___',
            '___w____'
        ]

        The top-most black disc can chain two jumps and eat both left white
        discs or jump only over the right white disc. The other black disc
        cannot move because it does produces any capturing move.

        The output must thus be:
        >> allowed_moves(board, 'b')
        [
            [(1, 2), (3, 0), (5, 2)],
            [(1, 2), (3, 4)]
        ]
    """
    allowed_capturing = []
    allowed_non_capturing = []
    size = len(board)
    disc_type = 1 if color == 'b' else -1
    nboard = board if isinstance(board, np.ndarray) else board_to_numpy(board)
    for i in range(size):
        for j in range(size):
            if nboard[i][j] != 0 and np.sign(nboard[i][j]) == disc_type:
                for ret in depth_first_search(nboard, nboard[i][j], i, j):
                    if ret['capturing']:
                        allowed_capturing.append(ret['capturing'])
                    if ret['non_capturing']:
                        allowed_non_capturing.append(ret['non_capturing'])
    return allowed_capturing if allowed_capturing else allowed_non_capturing

def alphabeta_play(board, color, eval_fn, max_depth):
    """ alpha beta pruning """
    def max_value(board, color, alpha, beta, eval_fn, depth):
        maxvalue = -np.inf
        maxmove = None
        if depth == 0:
            return None, eval_fn(board)
        moves = allowed_moves(board, color)
        if not moves:
            return None, eval_fn(board)
        for move in moves:
            newboard = apply_move(board, move)
            _, tempvalue = min_value(newboard, 'w', alpha, beta, eval_fn, depth-1)
            if tempvalue > maxvalue:
                maxmove, maxvalue = move, tempvalue
            elif tempvalue == maxvalue and random.choice([True, False]):
                maxmove = move

            if maxvalue >= beta:
                break
            alpha = max(alpha, maxvalue)
        if maxmove is None:
            maxvalue = eval_fn(board)
        return maxmove, maxvalue

    def min_value(board, color, alpha, beta, eval_fn, depth):
        minvalue = np.inf
        minmove = None
        if depth == 0:
            return None, eval_fn(board)
        moves = allowed_moves(board, color)
        if not moves:
            return None, eval_fn(board)
        for move in moves:
            newboard = apply_move(board, move)
            _, tempvalue = max_value(newboard, 'b', alpha, beta, eval_fn, depth-1)
            if tempvalue < minvalue:
                minmove, minvalue = move, tempvalue
            elif tempvalue == minvalue and random.choice([True, False]):
                minmove = move

            if minvalue <= alpha:
                break
            beta = min(beta, minvalue)
        if minmove is None:
            minvalue = eval_fn(board)
        return minmove, minvalue

    if color == 'b':
        best_move, best_value = max_value(board, 'b', -np.inf, np.inf, eval_fn, max_depth)
    elif color == 'w':
        best_move, best_value = min_value(board, 'w', -np.inf, np.inf, eval_fn, max_depth)
    return best_move, best_value

def random_play(board, color):
    """
        An example of play function based on allowed_moves.
    """
    moves = allowed_moves(board, color)
    # There will always be an allowed move
    # because otherwise the game is over and
    # 'play' would not be called by main.py
    return random.choice(moves)
