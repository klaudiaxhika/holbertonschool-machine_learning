#!/usr/bin/env python3
"""
conv_backward
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    conv_backward
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2) + 1
        pw = ((((w_prev - 1) * sw) + kw - w_prev) // 2) + 1
    else:
        return

    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)

    dA_prev = np.zeros((m, h_prev + (2 * ph), w_prev + (2 * pw), c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))

    for ex in range(m):
        for kernel_index in range(c_new):
            for h in range(h_new):
                for w in range(w_new):
                    i = h * sh
                    j = w * sw
                    dA_prev[ex, i: i + kh, j: j + kw, :] += (
                        dZ[ex, h, w, kernel_index] * W[:, :, :, kernel_index])
                    dW[:, :, :, kernel_index] += (
                        padded[ex, i: i + kh, j: j + kw, :] *
                        dZ[ex, h, w, kernel_index])

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
