#!/usr/bin/env python3
"""Import numpy"""

import numpy as np

def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1 or not isinstance(classes, int) or classes <= 0 \
        or classes < np.max(Y):
        return None
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot
