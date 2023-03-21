#!/usr/bin/env python3
"""Imports numpy"""

import numpy as np
"""Performs a same convolution on grayscale images"""


def convolve(images, kernels, padding='same', stride=(1, 1)):
    # Get dimensions of input images and kernels
    input_shape = images.shape
    kernel_shape = kernels.shape
    
    # Check that input is valid
    if input_shape[-1] != kernel_shape[-2]:
        raise ValueError("Last dimension of input images must match second-to-last dimension of kernels.")
    
    # Determine padding values
    if padding == 'same':
        # Calculate amount of padding required to preserve input shape
        pad_h = int(np.ceil(((input_shape[1] - 1) * stride[0] + kernel_shape[0] - input_shape[1]) / 2))
        pad_w = int(np.ceil(((input_shape[2] - 1) * stride[1] + kernel_shape[1] - input_shape[2]) / 2))
    elif padding == 'valid':
        pad_h = pad_w = 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'.")
    
    # Pad input images
    images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
    
    # Determine output shape
    output_h = int(np.floor((input_shape[1] + 2 * pad_h - kernel_shape[0]) / stride[0] + 1))
    output_w = int(np.floor((input_shape[2] + 2 * pad_w - kernel_shape[1]) / stride[1] + 1))
    output_shape = (input_shape[0], output_h, output_w, kernel_shape[-1])
    
    # Initialize output array
    output = np.zeros(output_shape)
    
    # Perform convolution for each kernel
    for i in range(kernel_shape[-1]):
        kernel = kernels[..., i]
        for j in range(output_h):
            for k in range(output_w):
                output[:, j, k, i] = np.sum(images[:, j*stride[0]:j*stride[0]+kernel_shape[0],
                                                      k*stride[1]:k*stride[1]+kernel_shape[1], :] * kernel, axis=(1, 2, 3))
    
    return output
