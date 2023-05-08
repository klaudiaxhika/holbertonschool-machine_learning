#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def precision(confusion):
    """
    Calculates a precision matrix
    """
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp

    return tp / (tp + fp)
