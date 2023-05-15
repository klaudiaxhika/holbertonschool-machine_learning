#!/usr/bin/env python3
"""Imports tensorflow"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    dropout_create_layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=dropout)
    return tensor(prev)
