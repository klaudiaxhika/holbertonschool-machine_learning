#!/usr/bin/env python3
"""
Defines class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor
        parameters:
            model_path: the path to where a Darknet Keras model is stored
            classes_path: the path to where the list of class names
                used for the Darknet model can be found
            class_t: represents the box score threshold for
                the initial filtering step
            nms_t: represents the IOU threshold for non-max suppression
            anchors: contains all the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            lines = f.readlines()
            self.class_names = []
            for name in lines:
                self.class_names.append(name[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Returns the output after passing through Sigmoid function
        output will be between 0 and 1
        """
        return (1. / (1. + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs
        """
        return None

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Determines filtered bounding boxes from processed outputs
        """
        return None

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses all non-max filter boxes to return predicted bounding box
        """
        return None

    @staticmethod
    def load_images(folder_path):
        """
        Loads images

        parameters:
            folder_path [str]: path to the folder holding all images to load

        returns:
            tuple of (images, image_paths):
                images [list]: images as numpy.ndarrays
                image_paths [list]: paths to the individual images
        """
        image_paths = glob.glob(folder_path + "/*")
        images = []
        for image in image_paths:
            images.append(cv2.imread(image))
        return (images, image_paths)
