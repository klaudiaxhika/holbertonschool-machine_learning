#!/usr/bin/env python3
"""
Update Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
    A class that represents a Gaussian process
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
        Calculates the covariance Kernel matrix between two matrices
        """
        first = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        sqdist = first + second - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y).reshape(X_s.shape[0])
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        std_s = np.diagonal(cov_s)
        return mu_s, std_s

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        X_new: numpy.ndarray of shape (1,) that represents the new
        sample point
        Y_new: numpy.ndarray of shape (1,) that represents the new
        sample function value
        No return just update
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
