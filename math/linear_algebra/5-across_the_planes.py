#!/usr/bin/env python3
"""a function that returns the sum of a matrix"""


def add_matrices2D(mat1, mat2):
    """returns sum  of a matrix"""
    result = []
    if [len(mat1), len(mat1[0])] != [len(mat2), len(mat2[0])]:
        return None
    for i in range(len(mat1)):
        result.append([])
        for j in range(len(mat1)):
            result[i].append(mat1[i][j]+mat2[i][j])
    return result
