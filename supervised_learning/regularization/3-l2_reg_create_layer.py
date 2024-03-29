#!/usr/bin/env python3
"""Imports tensorflow"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    l2_reg_create_layer
    """
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)

    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_initializer,
        kernel_regularizer=l2_reg)

    return (layer(prev))
