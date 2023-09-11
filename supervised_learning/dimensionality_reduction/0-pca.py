#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    # Calculate the covariance matrix of the data
    cov_matrix = np.cov(X, rowvar=False)
    
    # Perform eigen decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate the cumulative explained variance
    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    # Determine the number of principal components to keep
    num_components = np.argmax(explained_variance_ratio >= var) + 1
    
    # Select the top 'num_components' eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]
    
    # Calculate the weights matrix
    weights_matrix = top_eigenvectors
    
    return weights_matrix
