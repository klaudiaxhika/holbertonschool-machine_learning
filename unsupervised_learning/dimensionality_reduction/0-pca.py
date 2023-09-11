#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""function docs"""


def pca(X, var=0.95):
    """function docs"""
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    num_components = np.argmax(explained_variance_ratio >= var) + 1

    top_eigenvectors = eigenvectors[:, :num_components]

    weights_matrix = top_eigenvectors
    
    return weights_matrix

