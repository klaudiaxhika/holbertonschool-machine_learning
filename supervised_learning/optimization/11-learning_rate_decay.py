#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    learning_rate_decay
    """
    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return updated_alpha
