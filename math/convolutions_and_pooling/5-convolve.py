#!/usr/bin/env python3
"""Imports numpy"""

import numpy as np
"""Performs a same convolution on grayscale images"""


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolution on grayscale images
    """
    m, height, width, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if (padding == 'same'):
        ph = ((height - 1) * sh + kh - height) // 2 + 1
        pw = ((width - 1) * sw + kw - width) // 2 + 1
    elif (padding == 'valid'):
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    ch = ((height + 2*ph - kh) // sh) + 1
    cw = ((width + 2*pw - kw) // sw) + 1
    img_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    convoluted = np.zeros((m, ch, cw, c))
    for k in range(nc):
        i = 0
        for h in range(0, height + 2*ph - kh + 1, sh):
            j = 0
            for w in range(0, width + 2*pw - kw + 1, sw):
                output = np.sum(img_padded[:, h: h + kh, w: w + kw, :]
                                * kernels[:, :, :, k], axis=(1, 2))
                convoluted[:, i, j, :] = output
                j += 1
            i += 1
    return convoluted
