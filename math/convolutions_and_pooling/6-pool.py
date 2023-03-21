#!/usr/bin/env python3
"""Imports numpy"""

import numpy as np
"""Performs a same convolution on grayscale images"""


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on grayscale images
    """
    m, height, width, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
        
        
    pool_h = ((height - kh) // sh) + 1
    pool_w = ((width - kw) // sw) + 1
    
    pooled = np.zeros((m, pool_h, pool_w, c))
    
    i = 0
    for h in range(0, height - kh + 1, sh):
        j = 0
        for w in range(0, width - kw + 1, sw):
            if mode == 'max':
                output = np.max(images[:, h + kh, w: w + kw, :], axis=(1, 2))
            if mode == 'avg':
                output = np.average(images[:, h + kh, w: w + kw, :], axis=(1, 2))
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
