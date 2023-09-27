#!/usr/bin/env python3


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    update_variables_RMSProp
    """
    s_dW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (s_dW ** (1 / 2))))
    return var, s_dW
