#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np
"""Calculates a confusion matrix"""


def specificity(confusion):
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = confusion.sum() - (tp + fp + fn)

    return tn / (tn + fp)
