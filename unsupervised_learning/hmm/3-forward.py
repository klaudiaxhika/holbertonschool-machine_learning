#!/usr/bin/env python3
"""
A function that performs the forward algorithm
for a hidden markov model:
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    A function that performs the forward algorithm
    for a hidden markov model
    Observation: a numpy.ndarray of shape (T,) that contains
    the index of the observation
    Emission: a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Transition: a 2D numpy.ndarray of shape (N, N) containing
    the transition probabilities
    Initial: a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    Returns: P, F, or None, None on failure
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    sum_test = np.sum(Emission, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Transition, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Initial, axis=0)
    if not (sum_test == 1.0).all():
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observation[col]])

    P = np.sum(alpha[:, -1])

    return P, alpha
