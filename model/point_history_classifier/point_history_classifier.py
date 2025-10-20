#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point History Classifier Module

Provides a TensorFlow Lite model wrapper for classifying finger gestures
based on the history of finger tip movements over time.
"""
import numpy as np
import tensorflow as tf


class PointHistoryClassifier(object):
    """
    Classify finger gestures from point history data.
    
    This classifier uses a TensorFlow Lite model to identify finger movement
    patterns based on a sequence of finger tip positions over time.
    
    Attributes:
        interpreter (tf.lite.Interpreter): TensorFlow Lite interpreter for the model
        input_details (list): Input tensor details
        output_details (list): Output tensor details
        score_th (float): Confidence threshold for classification
        invalid_value (int): Value to return when confidence is below threshold
    """
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        """
        Initialize the point history classifier.
        
        Args:
            model_path (str, optional): Path to the TensorFlow Lite model file.
                Defaults to 'model/point_history_classifier/point_history_classifier.tflite'.
            score_th (float, optional): Minimum confidence threshold for valid predictions.
                Predictions below this threshold return invalid_value. Defaults to 0.5.
            invalid_value (int, optional): Value to return when confidence is too low.
                Defaults to 0.
            num_threads (int, optional): Number of threads for inference.
                Defaults to 1.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        """
        Classify a finger gesture from point history data.
        
        Args:
            point_history (list): Preprocessed and normalized point history coordinates
                as a flat list of floats.
        
        Returns:
            int: The index of the predicted gesture class, or invalid_value if
                confidence is below the threshold.
        """
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
