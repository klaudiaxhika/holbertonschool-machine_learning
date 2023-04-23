#!/usr/bin/env python3
"""Import numpy"""

import numpy as np

def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """
    try:
        m = Y.shape[0]
    except AttributeError:
        print("Error: Y should be a numpy.ndarray")
        return None

    if len(Y.shape) != 1:
        print("Error: Y should be a 1D array")
        return None

    if not np.issubdtype(Y.dtype, np.integer):
        print("Error: Y should contain only integers")
        return None

    if classes <= 0:
        print("Error: classes should be a positive integer")
        return None

    one_hot = np.zeros((classes, m))
    for i in range(m):
        if Y[i] < 0 or Y[i] >= classes:
            print("Error: label value out of range")
            return None
        one_hot[Y[i], i] = 1

    return one_hot