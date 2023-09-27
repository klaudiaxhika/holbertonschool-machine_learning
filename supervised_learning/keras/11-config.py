#!/usr/bin/env python3
"""Imports keras"""

import tensorflow.keras as K


def save_config(network, filename):
    """saves config"""
    json = network.to_json()
    with open(filename, 'w+') as f:
        f.write(json)
    return None


def load_config(filename):
    """loads config"""
    with open(filename, 'r') as f:
        json_string = f.read()
    model = K.models.model_from_json(json_string)
    return model
