#!/usr/bin/env python3
"""A function that initializes all variables
required to calculate the P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """A function that initializes all variables
    required to calculate the P affinities in t-SNE
    X: a numpy.ndarray of shape (n, d) containing the dataset
    to be transformed by t-SNE
    n: the number of data points
    d: the number of dimensions in each point
    perplexity: the perplexity that all Gaussian distributions should have
    return: (D, P, betas, H)"""
    n, d = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    H = np.log2(perplexity)
    betas = np.ones((n, 1))
    return D, P, betas, H
