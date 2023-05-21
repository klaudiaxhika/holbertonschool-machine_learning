#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """trains model"""

    callback = []

    if early_stopping and validation_data:
        callback.append(
            K.callbacks.EarlyStopping(monitor='loss', patience=patience))

    def learning_rate(epoch):
        """learning_rate"""
        return (alpha / (1 + decay_rate * epoch))

    if learning_rate_decay and validation_data:
        callback.append(
            K.callbacks.LearningRateScheduler(learning_rate, verbose=1))

    if save_best:
        callback.append(
            K.callbacks.ModelCheckpoint(filepath=filepath,
                                        save_best_only=True))

    if callback == []:
        callback = None

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
