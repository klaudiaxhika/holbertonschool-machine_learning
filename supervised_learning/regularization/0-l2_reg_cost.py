#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    l2_reg = 0

    for i in range(1, L + 1):
        l2_reg += np.sum(np.square(weights["W" + str(i)]))

    l2_reg *= (lambtha / (2 * m))

    cost += l2_reg

    return cost
