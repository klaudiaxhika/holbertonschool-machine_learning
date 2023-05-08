#!/usr/bin/env python3

def create_confusion_matrix(labels, logits):
    return np.matmul(labels.transpose(), logits)
