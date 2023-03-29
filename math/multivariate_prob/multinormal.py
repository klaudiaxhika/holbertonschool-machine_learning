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
        return init
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = mean_cov(data.T)
        
    def mean_cov(X):
        """
        return mean cov
        """
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")
        n, d = X.shape
        if n < 2:
            raise ValueError("X must contain multiple data points")
        mean = np.mean(X, axis=0)
        cov = np.zeros((d, d))
        for i in range(n):
            cov += np.outer((X[i] - mean), (X[i] - mean))
            cov /= (n - 1)
        return mean.reshape(1, d), cov

    def pdf(self, x):
        """
        return the pdf 
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
