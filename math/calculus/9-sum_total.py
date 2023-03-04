#!/usr/bin/env python3
""" a function for sum"""


def summation_i_squared(m):
    """returns squared sum"""
    if isinstance(m, (int, float)):
        return int(m*(m+1)*(2*m+1)/6)
    return None
