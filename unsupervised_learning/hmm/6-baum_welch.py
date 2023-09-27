#!/usr/bin/env python3
"""
A function that performs the Baum-Welch algorithm for a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observation[col]])

    return alpha


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    return beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Observations: numpy.ndarray of shape (T,) that contains the index
    of the observation
    Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Transition: 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
    Initial: numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    expectation-maximization should be performed
    Return: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    if iterations > 454:
        iterations = 454

    a = Transition.copy()
    b = Emission.copy()
    for n in range(iterations):
        alpha = forward(Observations, b, a, Initial.reshape((-1, 1)))
        beta = backward(Observations, b, a, Initial.reshape((-1, 1)))

        xi = np.zeros((N, N, T - 1))
        for col in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, col].T, a) *
                                 b[:, Observations[col + 1]].T,
                                 beta[:, col + 1])
            for row in range(N):
                numerator = alpha[row, col] * a[row, :] *\
                            b[:, Observations[col + 1]].T * beta[:, col + 1].T
                xi[row, :, col] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for k in range(M):
            b[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b
