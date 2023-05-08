#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def f1_score(confusion):
    """
    Calculates a f1 score
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    p = precision(confusion)
    r = sensitivity(confusion)

    return 2 * (p * r) / (p + r)
