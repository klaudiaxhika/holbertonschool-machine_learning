#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""class MultiNormal that represents a Multivariate Normal distribution"""


class MultiNormal:
    """
    class that represents MultiNormal distribution
    """

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

    def pdf(self, x):
        """
        Calculate the probability density function (PDF)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x must have the shape (" + str(self.mean.shape[0]) + ", 1)")
        else:
            # Compute the PDF
            D = np.sqrt(np.diag(C))
            D_inverse = 1 / np. outer (D, D)
            corr = D inverse * C
            return corr
