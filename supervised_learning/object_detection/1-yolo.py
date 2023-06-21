#!/usr/bin/env python3
"""
yolo class
"""

import numpy as np
import tensorflow.keras as keras

class Yolo:
    """
    yolo class
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = keras.models.load_model(model_path)
        self.class_names = self.load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_class_names(self, file_path):
        with open(file_path, 'r') as file:
            class_names = file.readlines()
        class_names = [name.strip() for name in class_names]
        return class_names

    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, num_anchors, _ = output.shape

            # Extract box coordinates, confidences, and class probabilities
            box = output[:, :, :, :4]
            box_confidence = output[:, :, :, 4:5]
            box_class_prob = output[:, :, :, 5:]

            # Reshape box coordinates to (grid_height, grid_width, num_anchors, 4)
            box = box.reshape((grid_height, grid_width, num_anchors, 4))

            # Calculate box coordinates relative to the original image size
            box[..., 0:2] = sigmoid(box[..., 0:2]) + create_meshgrid(grid_width, grid_height, self.anchors)
            box[..., 0:2] /= grid_width
            box[..., 2:4] = np.exp(box[..., 2:4]) * self.anchors / self.model.input.shape[1].value
            box[..., 2:4] /= image_size[::-1]

            # Append processed outputs to the respective lists
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def create_meshgrid(width, height, anchors):
    x = np.tile(np.arange(0, width), height)
    y = np.repeat(np.arange(0, height), width)

    return np.stack((x, y), axis=-1).reshape((1, height, width, 1, 2))
