#!/usr/bin/env python3
"""
A function that determines the probability of a markov chain
being in a particular state after a specific number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    A function that determines the probability of a markov chain
    being in a particular state after a specific number of iterations
    P: a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
      P[i, j]: the probability of transitioning from state i to state j
      n: the number of states in markov chain
    s: a numpy.ndarray of shape (1, n) representing the probability of
    starting in each state
    t: the number of iterations that the markov chain has been through
    returns: a numpy.ndarray of shape (1, n) representing the probability
    of being in a specific state after t iterations, or None on failure
    """
    if type(P) is not np.ndarray:
        return None
    if len(P.shape) != 2:
        return None
    n, n_t = P.shape
    if n != n_t:
        return None
    if type(s) is not np.ndarray:
        return None
    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != n:
        return None

    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    sn_t = s
    sn = np.zeros((1, n))
    for i in range(t):
        sn = np.matmul(sn_t, P)
        sn_t = sn

    return sn
