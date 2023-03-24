#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""calculates a correlation matrix"""

def correlation(C):
    """
    calculates a correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C) < 2 or (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")
    
    D = np.sqrt(np.diag(C))
    correlation_matrix = np.outer(D, D)
    correlation_matrix[C != 0] = C[C != 0] / correlation_matrix[C != 0]
    return correlation_matrix
