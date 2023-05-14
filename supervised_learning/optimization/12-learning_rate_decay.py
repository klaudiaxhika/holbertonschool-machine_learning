#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    learning_rate_decay
    """
    alpha_new = tf.divide(alpha, tf.pow(1 + decay_rate * tf.floor_div(global_step, decay_step), 1))
    return alpha_new
