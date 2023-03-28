#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""calculates a likelihood"""


def likelihood(x, n, P):
    """
    calculates the likelihood
    """
    factorial = np.math.factorial
    fact_coeff = factorial(n) / (factorial(n-x) * factorial(x))
    likelihood = fact_coeff * (P**x) * ((1 - P) ** (n -X))
    return likelihood
