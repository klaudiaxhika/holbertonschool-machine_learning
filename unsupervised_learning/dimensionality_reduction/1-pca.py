#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""function docs"""


def pca(X, ndim):
    """
    function docs
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    top_eigenvectors = eigenvectors[:, :ndim]

    transformed_data = np.dot(X_centered, top_eigenvectors)

    return transformed_data
