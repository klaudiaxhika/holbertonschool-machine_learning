#!/usr/bin/env python3
"""
Defines the class LSTMCell that represents an LSTM unit
"""
import numpy as np


class LSTMCell:
    """
    Represents a LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Performs the softmax function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Performs the sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        parameters:
            h_prev: contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
                i: dimensionality of the data
            c_prev: contains previous cell state
            x_t: contains data input for the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        update_gate = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        intermediate_cell_state = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        output_gate = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        c_next = forget_gate * c_prev + update_gate * intermediate_cell_state
        h_next = output_gate * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
