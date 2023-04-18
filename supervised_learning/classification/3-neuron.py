#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""class that represents a single neuron performing binary classification"""


class Neuron:
    """
    class that represents a single neuron performing binary classification
    """

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return (self.__W)

    @property
    def b(self):
        return (self.__b)

    @property
    def A(self):
        return (self.__A)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression function:
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)
