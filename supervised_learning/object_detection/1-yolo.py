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
            grid_height, grid_width, num_boxes, _ = output.shape

            # Process boundary boxes
            box = output[..., :4]
            box[..., 0:2] = 1 / (1 + np.exp(-box[..., 0:2]))
            box[..., 2:4] = np.exp(box[..., 2:4]) * self.anchors / self.model.input.shape[1:3]

            # Adjust boundary box coordinates relative to original image size
            image_height, image_width = image_size
            box[..., 0:1] *= image_width
            box[..., 1:2] *= image_height
            box[..., 2:3] *= image_width
            box[..., 3:4] *= image_height

            boxes.append(box)

            # Extract box confidences and class probabilities
            box_confidence = output[..., 4:5]
            box_confidences.append(box_confidence)

            class_probs = output[..., 5:]
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
