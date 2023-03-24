#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""class MultiNormal that represents a Multivariate Normal distribution"""


class MultiNormal:
    def __init__(self, data):
        """
        Initialize a MultiNormal instance with the given data set
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n = data.shape[1]
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean

        self.cov = np.matmul(centered, centered.T) / (n - 1)
