#!/usr/bin/env python3
"""imports numpy"""


import numpy as np

"""function to add matrices"""


def add_matrices(mat1, mat2):
    """returns matrix sum"""
    if np.shape(mat1) != np.shape(mat2):
        return None
    else:
        return np.add(mat1, mat2);
