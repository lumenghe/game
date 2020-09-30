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
