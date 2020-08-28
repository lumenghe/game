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
