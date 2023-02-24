#!/usr/bin/env python3
""" a function that returns matrix multiplication"""


def mat_mul(mat1, mat2):
    """returns multiplicated matrix"""
    if len(mat1[0]) != len(mat2):
        return None
    mat = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))] 
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            product = 0
            for k in range(len(mat2)):
                mat[i][j] += mat1[i][k] * mat2[k][j]
    return mat
