#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""class that represents a neural network with one hidden layer"""


class NeuralNetwork:
    """
    class that represents a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # nodes number of nodes and receives nx number of input feature
        self.W1 = np.random.randn(nodes, nx)
        # matches the shape of the hidden layer output
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # only one node and receives the output from the hidden layer as input
        self.W2 = np.random.randn(1, nodes)
        # matching the shape of the output neuron's activation
        self.b2 = 0
        self.A2 = 0
