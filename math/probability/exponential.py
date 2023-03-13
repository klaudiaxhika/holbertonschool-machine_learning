#!/usr/bin/env python3
"""a class Exponential that represents an exponential distribution"""


class Exponential:
    """
    a class Exponential that represents an exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1/(float(sum(data)) / len(data))

    def pdf(self, x):
        """
        a class Exponential that represents an exponential distribution
        """
        if type(x) is not int:
            x = int(x)
        if (x < 0):
            return 0
        e = 2.7182818285
        pdf = self.lambtha * (e ** (-self.lambtha * x))
        return pdf

    def cdf(self, x):
        if type(x) is not int:
            x = int(x)
        if (x < 0):
            return 0
        e = 2.7182818285
        cdf = 1 - (e ** (-self.lambtha * x))
        return cdf
