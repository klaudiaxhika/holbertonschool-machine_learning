#!/usr/bin/env python3
"""
a function that creates the training operation for the network
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Args:
        loss (tf.Tensor): the loss of the network's prediction.
        alpha (float): the learning rate.

    Returns:
        tf.Operation: an operation that trains the network
        using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
