#!/usr/bin/env python3
""" a function that calculates the derivative"""


def poly_derivative(poly):
    """returns the coeficents"""
    der_list = []
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    for power, coeff in enumerate(poly):
        if(power == 0 and len(poly) > 1):
            continue
        der_list.append(power * coeff)
    return der_list
