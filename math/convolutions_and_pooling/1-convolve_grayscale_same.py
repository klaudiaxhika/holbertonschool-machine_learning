#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same  convolution on grayscale images
    """
    m, height, width = images.shape
    kh, kw = kernel.shape
    img_padded = np.pad(images, ((0,0), (1,1), (1,1)), 'constant')
    convoluted = np.zeros((m, height, width))
    for h in range(height - kh + 1):
        for w in range(width - kw + 1):
            output = np.sum(img_padded[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
