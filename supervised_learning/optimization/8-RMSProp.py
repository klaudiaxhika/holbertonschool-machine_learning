#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    create_RMSProp_op
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
