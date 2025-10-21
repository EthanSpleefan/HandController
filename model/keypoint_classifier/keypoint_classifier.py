#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keypoint Classifier Module

Provides a TensorFlow Lite model wrapper for classifying hand gestures
based on hand landmark keypoints detected by MediaPipe.
"""
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    """
    Classify hand gestures from preprocessed landmark keypoints.

    This classifier uses a TensorFlow Lite model to identify hand gestures
    based on normalized hand landmark positions.

    Attributes:
        interpreter (tf.lite.Interpreter): TensorFlow Lite interpreter for the model
        input_details (list): Input tensor details
        output_details (list): Output tensor details
    """
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        """
        Initialize the keypoint classifier.

        Args:
            model_path (str, optional): Path to the TensorFlow Lite model file.
                Defaults to 'model/keypoint_classifier/keypoint_classifier.tflite'.
            num_threads (int, optional): Number of threads for inference.
                Defaults to 1.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        """
        Classify a hand gesture from landmark keypoints.

        Args:
            landmark_list (list): Preprocessed and normalized landmark coordinates
                as a flat list of floats.

        Returns:
            int: The index of the predicted gesture class.
        """
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
