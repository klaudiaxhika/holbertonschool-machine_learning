#!/usr/bin/env python3
"""
Imports numpy
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    update_variables_momentum
    """
    dW_prev = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * dW_prev)
    return var, dW_prev
