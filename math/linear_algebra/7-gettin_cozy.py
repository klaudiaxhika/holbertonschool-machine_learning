#!/usr/bin/env python3
""" a function that returns concat of matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """returns concat matrix"""
    result = []
    if (axis == 0):
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    if(axis == 1):
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])

    return result
