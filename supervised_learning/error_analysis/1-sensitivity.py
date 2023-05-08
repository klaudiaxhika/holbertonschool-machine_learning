#!/usr/bin/env python3
"""Imports numpy"""

import numpy as np

"""Calculates a sensitivity matrix"""


def sensitivity(confusion):
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp

    return tp / (tp + fn)
