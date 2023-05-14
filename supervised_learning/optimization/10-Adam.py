#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    create_Adam_op
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha, 
        beta_1=beta1, 
        beta_2=beta2, 
        epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
