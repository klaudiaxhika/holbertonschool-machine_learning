#!/usr/bin/env python3
"""Import keras"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves weights"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads weights"""
    network.load_weights(filename)
    return None
