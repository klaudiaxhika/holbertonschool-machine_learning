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
            raise ValueError("x must have the shape ({d}, 1)")
       
        # Compute the PDF
        d = self.mean.shape[0]
        centered = x - self.mean
        exponent = -0.5 * np.matmul(centered.T, np.matmul(np.linalg.inv(self.cov), centered))
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        return norm_const * np.exp(exponent)
