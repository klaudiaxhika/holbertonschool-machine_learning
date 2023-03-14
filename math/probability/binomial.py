#!/usr/bin/env python3
"""a class Binomial that represents a binomial distribution"""


class Binomial:
    """
    a class Binomial that represents a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        self.n = int(n)
        self.p = float(p)
        if n <= 0:
            raise ValueError("n must be a positive value")
        if not 0 => p < 1:
            raise ValueError("p must be greater than 0 and less than 1")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
