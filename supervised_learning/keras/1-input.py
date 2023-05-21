#!/usr/bin/env python3
"""imports keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds model"""
    reg_l2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    X = inputs

    for i in range(len(layers)):
        if i != 0:
            X = K.layers.Dropout(1 - keep_prob)(X)
        X = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=reg_l2)(X)
        
    model = K.Model(inputs=inputs, outputs=X)
    return model
