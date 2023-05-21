#!/usr/bin/env python3
"""Imports keras"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """prediction function"""
    prediction = network.predict(x=data,
                                 verbose=verbose)
    return prediction
