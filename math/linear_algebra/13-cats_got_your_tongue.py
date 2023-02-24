#!/usr/bin/env python3
"""imports numpy library"""


import numpy as np

"""a function that concats 2 matrix"""


def np_cat(mat1, mat2, axis=0):
    """returns concatenated matrix"""
    return np.concatenate((mat1, mat2), axis)
