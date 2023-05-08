#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np
"""Calculates a confusion matrix"""


def create_confusion_matrix(labels, logits):
    return np.matmul(labels.transpose(), logits)
