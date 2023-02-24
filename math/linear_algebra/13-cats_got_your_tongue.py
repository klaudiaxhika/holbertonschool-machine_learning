#!/usr/bin/env python3
"""a function that concats 2 matrix"""


import numpy as np

"""a function that concats 2 matrix"""


def np_cat(mat1, mat2, axis=0):
    """returns concatenated matrix"""
    return np.concatenate((mat1, mat2), axis)
