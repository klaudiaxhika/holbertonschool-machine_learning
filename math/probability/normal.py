#!/usr/bin/env python3
"""a class Normal that represents a normal distribution"""


class Normal:
    """
    a class Normal that represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if self.stddev <= 0:
            raise ValueError("stddev must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            sumStddev = 0
            for i in data:
                sumStddev += (i - self.mean) ** 2
            self.stddev = float(sumStddev / len(data))
