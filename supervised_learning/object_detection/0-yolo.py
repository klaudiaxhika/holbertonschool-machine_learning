#!/usr/bin/env python3
"""
class Yolo
"""


import numpy as np
import tensorflow.keras as keras
"""
class Yolo
"""


class Yolo:
    """
    class Yolo
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

    def detect_objects(self, image):
        # Implement object detection using the YOLO v3 algorithm
        # Return the detected objects with their bounding boxes,
        # class labels, and confidence scores
        pass
