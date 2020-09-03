import os
import sys
import json
import copy
import requests

import ai

url_prefix = 'https://www.xxxxx.com/checkers/'

###############################################################################

class InvalidMoveException(Exception):
    pass

class GameOver(Exception):
    def __init__(self, winner):
        super(GameOver, self).__init__()
        self.winner = winner

###############################################################################

if sys.version_info >= (3,0):
    def raw_input(msg):
        return input(msg)
else:
    pass

###############################################################################

def send_request(url, method, data = {}):
    if method == 'get':
        response = requests.get(url, params = data)
    elif method == 'post':
        response = requests.post(url, data = data)
    else:
        raise Exception('Bad method')

    try:
        content = json.loads(response.content.decode('utf-8'))
    except:
        raise Exception('Could not decode response')

    if response.status_code == 200:
        if 'error' in content:
            if content['error'] == 'InvalidMove':
                raise InvalidMoveException()
            else:
                raise Exception(content['error'])
        else:
            return content
    else:
        if 'error' in content:
            raise Exception(content['error'])
        else:
            raise Exception('Unknown error')

###############################################################################

def print_board(board):
    b = "_" * (2 * len(board) + 3) + "\n"
    for i, row in enumerate(board):
        b += "| "
        for j, c in enumerate(row):
            if (i + j) % 2 == 0:
                b += "  "
            else:
                b += c + " "
        b += "|\n"
    b += "-" * (2 * len(board) + 3)
    print(b + "\n")

###############################################################################

def print_move(msg, move):
    print(msg + " " + ", ".join([str(tuple(m)) for m in move]))

###############################################################################

def new_game(config, size, color):
    data = copy.deepcopy(config)
    data['size'] = size
    data['color'] = color
    return send_request(url_prefix + 'games', 'post', data)

###############################################################################

def new_move(game, move):
    # Check the move
    assert(len(move) > 0)
    mm = []
    for m in move:
        assert(len(m) == 2)
        mm.append(list(m))

    print_move("Your move:", mm)
    response = send_request(url_prefix + 'games/%d' % game['id'], 'post', {'move': mm})

    if 'board_after_candidate_move' in response:
        print_board(response['board_after_candidate_move'])
    if 'move' in response:
        print_move("The other side made this move:", response['move'])
    if response['over']:
        if 'board' in response:
            print_board(response['board'])
        raise GameOver(response['winner'])

    return response['board']

###############################################################################
