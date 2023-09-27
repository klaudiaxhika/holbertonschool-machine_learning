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
                used for the Darknet model can be found,
                list is ordered by order of index
            class_t: represents the box score threshold for
                the initial filtering step
            nms_t: represents the IOU threshold for non-max suppression
            anchors:contains all the anchor boxes
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
        """
        image_paths = glob.glob(folder_path + "/*")
        images = []
        for image in image_paths:
            images.append(cv2.imread(image))
        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Resizes and rescales the images before processeing
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value

        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(image, dsize=(input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255
            pimages.append(rescaled)
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores
        """
        for i, box in enumerate(boxes):
            x, y, w, h = box
            class_index = box_classes[i]
            class_name = self.class_names[class_index]
            score = box_scores[i]

            # Draw box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Write class name
            text = f"{class_name} {score:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)

        while True:
            key = cv2.waitKey(0)
            if key == ord('s'):  # Save the image
                os.makedirs("detections", exist_ok=True)
                save_path = os.path.join("detections", file_name)
                cv2.imwrite(save_path, image)
                break
            else:  # Close the image window without saving
                cv2.destroyAllWindows()
                break


    def predict(self, folder_path):
        """
        Predicts objects in images from the given folder
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        predictions = []

        for i, pimage in enumerate(pimages):
            image = images[i]
            image_shape = image_shapes[i]
            file_name = os.path.basename(image_paths[i])

            # Perform object detection
            outputs = self.model.predict(np.expand_dims(pimage, axis=0))
            boxes, box_classes, box_scores = self.process_outputs(outputs, image_shape)
            self.show_boxes(image, boxes, box_classes, box_scores, file_name)
            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths
