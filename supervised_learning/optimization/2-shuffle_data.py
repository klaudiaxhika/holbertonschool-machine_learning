#!/usr/bin/env python3
"""
Imports numpy
"""


import numpy as np


def shuffle_data(X, Y):
    """
    shuffle_data
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
