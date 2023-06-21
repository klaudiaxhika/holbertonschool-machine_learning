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
            grid_height, grid_width, anchor_boxes, _ = output.shape
            num_classes = output.shape[-1] - 5

            # Extract box coordinates, box confidences, and box class probabilities
            box_coords = output[..., :4]
            box_conf = output[..., 4:5]
            box_probs = output[..., 5:]

            # Apply sigmoid function to box confidences and box class probabilities
            box_conf = 1 / (1 + np.exp(-box_conf))
            box_probs = 1 / (1 + np.exp(-box_probs))

            # Scale box coordinates to the original image size
            grid_height_ratio = image_size[0] / grid_height
            grid_width_ratio = image_size[1] / grid_width
            anchor_boxes = len(self.anchors)

            box_coords[..., 0] = (box_coords[..., 0] + self.sigmoid(box_coords[..., 0])) * grid_width_ratio
            box_coords[..., 1] = (box_coords[..., 1] + self.sigmoid(box_coords[..., 1])) * grid_height_ratio
            box_coords[..., 2] = self.anchors[:, 0] * np.exp(box_coords[..., 2]) * grid_width_ratio
            box_coords[..., 3] = self.anchors[:, 1] * np.exp(box_coords[..., 3]) * grid_height_ratio

            # Append processed outputs to respective lists
            boxes.append(box_coords)
            box_confidences.append(box_conf)
            box_class_probs.append(box_probs)

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
