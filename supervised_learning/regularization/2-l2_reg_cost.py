#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf


def l2_reg_cost(cost, lam):
    """
    l2_reg_cost
    """
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)
