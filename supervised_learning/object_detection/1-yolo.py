#!/usr/bin/env python3
"""
yolo class
"""

import numpy as np
import tensorflow.keras as keras

class Yolo:
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

        # Extract bounding box coordinates
        box = output[..., :4]
        box[..., :2] = self.sigmoid(box[..., :2])
        box[..., 2:] = np.exp(box[..., 2:]) * self.anchors / np.array([grid_width, grid_height])
        box[..., 0] = (box[..., 0] - box[..., 2] / 2) * image_size[1]
        box[..., 1] = (box[..., 1] - box[..., 3] / 2) * image_size[0]
        box[..., 2] = (box[..., 0] + box[..., 2] / 2) * image_size[1]
        box[..., 3] = (box[..., 1] + box[..., 3] / 2) * image_size[0]

        # Extract confidence scores and class probabilities
        box_confidence = self.sigmoid(output[..., 4:5])
        class_probs = self.sigmoid(output[..., 5:])

        boxes.append(box)
        box_confidences.append(box_confidence)
        box_class_probs.append(class_probs)

    return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
