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
            self.stddev = float((sumStddev / len(data)) ** (1/2))

    def z_score(self, x):
        """
        calculates z-score
        """
        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
        calculates x-value
        """
        return float(z * self.stddev + self.mean)

    def pdf(self, x):
        """
        calculates pdf
        """
        e = 2.7182818285
        pi = 3.1415926536
        exponent = ((x - self.mean) ** 2) / (2 * (self.stddev) ** 2)
        y = 1 / (e ** exponent)
        return y / (self.stddev * (2 * pi) ** (1/2))

    def erf(self, x):
        return (x - (x ** 3/3) + (x ** 5/10) - (x ** 7/42) + (x ** 9/216))

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value
        """
        y = ((x - self.mean) / (self.stddev * (2 ** (1/2))))
        return (1/2) * (1 + self.erf(y))
