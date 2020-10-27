import main
import ai

def convert_board(size, board):
    board = board.replace('\n', '')
    board = board.replace(' ', '')
    return [board[i:i + size] for i in range(0, len(board), size)]

def uniform_moves(moves):
    return [[list(m) for m in mm] for mm in moves]

def check_moves(moves, ground_truth):
    moves = uniform_moves(moves)
    ground_truth = uniform_moves(ground_truth)
    ok = True
    for move in moves:
        if move not in ground_truth:
            print("FAILED: unexpected move: " + str(move))
            ok = False
    for move in ground_truth:
        if move not in moves:
            print("FAILED: missing move: " + str(move))
            ok = False
    if ok:
        print("OK")
    return ok

def test_01_move_black_disc():
    board = convert_board(8, """
________
________
________
__b_____
________
________
_____w__
________
""")
    ground_truth = [[(3, 2), (4, 1)],
                    [(3, 2), (4, 3)]]
    moves = ai.allowed_moves(board, 'b')
    return board, ground_truth, check_moves(moves, ground_truth)

def test_02_move_black_disc_border():
    board = convert_board(8, """
________
________
________
b_______
________
________
_____w__
________
""")
    ground_truth = [[(3, 0), (4, 1)]]
    moves = ai.allowed_moves(board, 'b')
    return board, ground_truth, check_moves(moves, ground_truth)

def test_03_move_white_disc():
    board = convert_board(8, """
________
________
________
__b_____
________
________
_____w__
________
""")
    ground_truth = [[(6, 5), (5, 6)],
                    [(6, 5), (5, 4)]]
    moves = ai.allowed_moves(board, 'w')
    return board, ground_truth, check_moves(moves, ground_truth)

def test_04_move_black_king():
    board = convert_board(8, """
________
________
________
__B_____
________
________
_____w__
________
""")
    ground_truth = [[(3, 2), (4, 1)],
                    [(3, 2), (4, 3)],
                    [(3, 2), (2, 1)],
                    [(3, 2), (2, 3)]]
    moves = ai.allowed_moves(board, 'b')
    return board, ground_truth, check_moves(moves, ground_truth)

def test_05_move_black_king():
    board = convert_board(8, """
________
________
________
__B_____
________
________
_____W__
________
""")
    ground_truth = [[(6, 5), (5, 6)],
                    [(6, 5), (5, 4)],
                    [(6, 5), (7, 6)],
                    [(6, 5), (7, 4)]]
    moves = ai.allowed_moves(board, 'w')
    return board, ground_truth, check_moves(moves, ground_truth)

def test_06_move_black_initial():
    board = convert_board(8, """
_b_b_b_b
b_b_b_b_
_b_b_b_b
________
________
w_w_w_w_
_w_w_w_w
w_w_w_w_
""")
    ground_truth = [[(2, 1), (3, 0)],
                    [(2, 1), (3, 2)],
                    [(2, 3), (3, 2)],
                    [(2, 3), (3, 4)],
                    [(2, 5), (3, 4)],
                    [(2, 5), (3, 6)],
                    [(2, 7), (3, 6)]]
    moves = ai.allowed_moves(board, 'b')
    return board, ground_truth, check_moves(moves, ground_truth)
