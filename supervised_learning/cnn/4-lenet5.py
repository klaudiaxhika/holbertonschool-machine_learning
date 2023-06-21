#!/usr/bin/env python3
"""
lenet5
"""

import tensorflow as tf

def lenet5(x, y):
    """
    lenet5
    """
    # Convolutional layer 1
    conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal')(x)
    # Max pooling layer 1
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Convolutional layer 2
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                                   activation='relu',
                                   kernel_initializer='he_normal')(pool1)
    # Max pooling layer 2
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Flatten the output from the previous layer
    flatten = tf.keras.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.keras.layers.Dense(120, activation='relu',
                                kernel_initializer='he_normal')(flatten)

    # Fully connected layer 2
    fc2 = tf.keras.layers.Dense(84, activation='relu',
                                kernel_initializer='he_normal')(fc1)

    # Output layer
    output = tf.keras.layers.Dense(10, activation='softmax')(fc2)

    # Define the loss function
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, output))

    # Define the accuracy metric
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, output))

    # Define the optimizer and training operation
    optimizer = tf.keras.optimizers.Adam()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
