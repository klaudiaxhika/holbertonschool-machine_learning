#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images
    """
    m, height, width = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    img_padded = np.pad(images, ((0,0), padding, padding), mode='constant')
    convoluted = np.zeros((m, height + 2*ph - kh + 1, width + 2*pw - kw + 1))
    for h in range(0, height - kh + 1, 1):
        for w in range(0, width - kw + 1, 1):
            output = np.sum(img_padded[:, h: h + kh, w: w + kw] * kernel,                             axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
