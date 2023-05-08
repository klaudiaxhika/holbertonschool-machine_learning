#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Returns a confusion matrix
    """
    return np.matmul(labels.transpose(), logits)
