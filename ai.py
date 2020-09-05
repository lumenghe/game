"""
    ai
"""
import random
import numpy as np
from model import ValueNet

def inboard(x, y, size=8):
    """ check if (x,y) in board or not """
    return (0 <= x < size) and (0 <= y < size)
