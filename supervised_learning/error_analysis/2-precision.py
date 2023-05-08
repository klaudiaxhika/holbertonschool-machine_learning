#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np
"""Calculates a precision matrix"""


def precision(confusion):
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp

    return tp / (tp + fp)
