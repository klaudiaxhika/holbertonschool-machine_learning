#!/usr/bin/env python3
"""Imports numpy"""

import numpy as np
"""Performs a same convolution on grayscale images"""


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution with padding on grayscale images
    """
    m, height, width = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    ch = height + 2*ph - kh + 1
    cw = width + 2*pw - kw + 1
    convoluted = np.zeros((m, height + 2*ph - kh + 1, width + 2*pw - kw + 1))
    for h in range(ch):
        for w in range(cw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
