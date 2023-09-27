#!/usr/bin/env python3
"""
Calculate the accuracy of a prediction for the neural network
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate the accuracy of a prediction for the neural network
    """
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, "float"))
    return accuracy
