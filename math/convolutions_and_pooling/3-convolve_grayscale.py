#!/usr/bin/env python3
import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution with stride on grayscale images
    """
    m, height, width = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if (type(padding) is list):
        ph, pw = padding
    elif (padding == 'same'):
        if (kh % 2) == 1 and (kw % 2) == 1:
            ph = (kh - 1) // 2
            pw = (kw - 1) // 2
        else:
            ph = kh // 2
            pw = kw // 2
    elif (padding == 'valid'):
        ph, pw = (0,0)
        
    ch = (height + 2*ph - kh + 1) // sh
    cw = (width + 2*pw - kw + 1) // sw
    img_padded = np.pad(images, ((0,0), (ph,ph), (pw,pw)), mode='constant')
    convoluted = np.zeros((m, ch, cw))
    
    for h in range(ch):
        for w in range(cw):
            output = np.sum(img_padded[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted

