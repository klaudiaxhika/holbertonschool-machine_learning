#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def sensitivity(confusion):
    """
    Returns a sensitivity matrix
    """
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp

    return tp / (tp + fn)
