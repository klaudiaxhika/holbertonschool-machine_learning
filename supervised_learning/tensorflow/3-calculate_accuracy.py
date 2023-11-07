#!/usr/bin/env python3
"""
A function  that calculates the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y -- tf.placeholder for the labels of the input data.
    y_pred -- tensor containing the network's predictions.

    Returns:
    Tensor containing the decimal accuracy of the prediction.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
