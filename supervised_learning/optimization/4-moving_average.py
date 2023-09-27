#!/usr/bin/env python3
"""
Imports numpy
"""


import numpy as np


def moving_average(data, beta):
    """
    shuffle_data
    """
    v = 0
    EMA = []
    for i in range(len(data)):
        v = ((v * beta) + ((1 - beta) * data[i]))
        EMA.append(v / (1 - (beta ** (i + 1))))
    return EMA
