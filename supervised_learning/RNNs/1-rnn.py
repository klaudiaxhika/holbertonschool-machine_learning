#!/usr/bin/env python3
"""
A function that performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    A function that performs forward propagation for a simple RNN
    parameters:
    rnn_cell: is an instance of RNNCell
    that will be used for the forward propagation
    X: is the data to be used, given as a numpy.ndarray
    of shape (t, m, i) where
    t: is the max number of time steps
    m: is the batch size
    i: is the dimensionality of the data
    h_0: is the initial hidden state given as a numpy.ndarray
    of shape (m, h) where
    h: is the dimenosionality of the hidden state
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)
