#!/usr/bin/env python3
"""a function that returns a transposed  matrix"""


def matrix_transpose(matrix):
    """returns a transposed  matrix"""
    return [
        [
            matrix[j][i] for j in range(len(matrix))
        ] for i in range(len(matrix[0]))
    ]
