#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""calculates a confusion matrix"""


def create_confusion_matrix(labels, logits):
    return np.matmul(labels.transpose(), logits)
