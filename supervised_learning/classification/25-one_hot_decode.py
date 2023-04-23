#!/usr/bin/env python3
"""Import numpy"""

import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    m = one_hot.shape[1]
    labels = np.argmax(one_hot, axis=0)
    return labels
