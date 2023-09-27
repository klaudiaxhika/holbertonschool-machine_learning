#!/usr/bin/env python3
"""
Defines the class GRUCell that represents a gated recurrent unit
"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        parameters:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """Performs the softmax function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Performs the sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        parameters:
            h_prev: contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t: contains data input for the cell
        """
        concatenation1 = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(concatenation1, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(concatenation1, self.Wr) + self.br)

        concatenation2 = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation2, self.Wh) + self.bh)
        h_next *= z_gate
        h_next += (1 - z_gate) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
