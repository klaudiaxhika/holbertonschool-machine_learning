#!/usr/bin/env python3
"""a function that returns sum of arrays"""


def add_arrays(arr1, arr2):
    """returns sum of arrays"""
    result = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        result.append(arr1[i]+arr2[i])
    return result
