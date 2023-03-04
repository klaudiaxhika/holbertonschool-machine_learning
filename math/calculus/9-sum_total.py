#!/usr/bin/env python3
"""imports numpy"""


import numpy as np
""" a function that for sum"""


def summation_i_squared(m):
"""returns squared sum"""
    if isinstance(m, int):
        return int(m*(m+1)*(2*m+1)/6)
    return None
