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
