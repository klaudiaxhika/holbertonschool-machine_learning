#!/usr/bin/env python3
"""Imports keras"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ test model """
    loss, accuracy = network.evaluate(x=data,
                                      y=labels,
                                      verbose=verbose)
    return loss, accuracy
