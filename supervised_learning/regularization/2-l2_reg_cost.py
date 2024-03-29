#!/usr/bin/env python3
"""A function that calculates the cost of a neural network
with L2 regularization"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    A function that calculates the cost of a neural network
    with l2 regularization"""
    return cost + tf.losses.get_regularization_losses()
