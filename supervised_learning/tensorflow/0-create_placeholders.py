#!/usr/bin/env python3
"""
the function that returns two placeholders, x and y, for the neural network
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for input and one-hot labels for the neural network.

    Arguments:
    nx -- integer, number of feature columns in our data
    classes -- integer, number of classes in our classifier

    Returns:
    x -- placeholder for the input data to the neural network
    y -- placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
