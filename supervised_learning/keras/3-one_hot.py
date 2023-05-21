#!/usr/bin/env python3
"""imports keras"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """one_hot"""
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
