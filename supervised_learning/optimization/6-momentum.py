#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    """
    create_momentum_op
    """
    optimizer = tf.train.MomentumOptimizer(alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
