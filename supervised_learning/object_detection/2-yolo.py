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
        # Implementation of process_outputs method
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(boxes, box_confidences, box_class_probs):
            # Compute box scores by multiplying box confidences and class probabilities
            box_scores_per_class = confidence * class_probs

            # Find the class index with the maximum box score
            class_index = np.argmax(box_scores_per_class, axis=-1)

            # Select box scores and boxes based on the box score threshold
            mask = box_scores_per_class >= self.class_t
            scores = box_scores_per_class[mask]
            selected_boxes = box[mask]

            # Apply non-max suppression to filter overlapping boxes
            indices = self.non_max_suppression(selected_boxes, scores)

            # Add filtered boxes, class indices, and box scores to the respective lists
            filtered_boxes.extend(selected_boxes[indices])
            box_classes.extend(class_index[mask][indices])
            box_scores.extend(scores[indices])

        # Convert lists to numpy arrays
        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, boxes, scores):
        # Implementation of non-max suppression method
        pass
