#!/usr/bin/env python3
"""a function that returns the sum of a matrix"""


def add_matrices2D(mat1, mat2):
    """returns sum  of a matrix"""
    mat = []
    if len(mat1[0]) != len(mat2[0]) or len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
         mat.append(row)
    return mat
