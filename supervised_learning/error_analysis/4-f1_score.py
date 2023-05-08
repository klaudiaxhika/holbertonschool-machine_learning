#!/usr/bin/env python3
"""Imports numpy"""


import numpy as np
"""Calculates a f1 score"""


def f1_score(confusion):
    p = precision(confusion)
    r = sensitivity(confusion)

    return 2 * (p * r) / (p + r)
