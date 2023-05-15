#!/usr/bin/env python3
"""Imports tensorflow"""


import tensorflow as tf


def l2_reg_cost(cost, lambtha):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    with tf.name_scope('l2_reg_cost'):
        # Get all the trainable variables in the graph
        trainable_vars = tf.trainable_variables()
        
        # Calculate the L2 regularization term for each variable
        l2_reg_terms = [tf.nn.l2_loss(var) for var in trainable_vars]
        
        # Add up the L2 regularization terms
        l2_reg = tf.add_n(l2_reg_terms)
        
        # Add the L2 regularization term to the cost
        cost_with_l2_reg = tf.add(cost, lambtha * l2_reg, name='cost_with_l2_reg')
        
        return cost_with_l2_reg
