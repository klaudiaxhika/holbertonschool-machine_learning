#!/usr/bin/env python3
"""A function that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """A function that tests a neural network"""
    return network.evaluate(data, labels, verbose=verbose)
