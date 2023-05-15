#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf


def l2_reg_cost(cost, lam):
    """
    l2_reg_cost
    """
    l2_reg_losses = sum(tf.nn.l2_loss(w) for w in model.trainable_variables)
    return cost + lam * l2_reg_losses
