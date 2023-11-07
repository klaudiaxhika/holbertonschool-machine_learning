#!/usr/bin/env python3
"""ctivation function that the layer should use"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network.

    Arguments:
    prev -- tensor output of the previous layer
    n -- integer, number of nodes in the layer to create
    activation -- activation function that the layer should use

    Returns:
    tensor output of the layer
    """
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    W = tf.Variable(initializer([prev.shape[1].value, n]), name='W')
    b = tf.Variable(tf.zeros([n]), name='b')
    Z = tf.matmul(prev, W) + b

    if activation is not None:
        A = activation(Z)
    else:
        A = Z

    return A
