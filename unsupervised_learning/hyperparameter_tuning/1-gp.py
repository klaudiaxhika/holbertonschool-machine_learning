#!/usr/bin/env python3
"""
A function that predicts the mean and standard deviation
of points in a Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    A class that represents the Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        A function that initializes a class
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        first = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        sqdist = first + second - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian process
        X_s: numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
        return: mu, sigma
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y).reshape(X_s.shape[0])

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var_s = np.diagonal(cov_s)

        return mu_s, var_s
