"""
FPS Calculator Module

Provides a utility class for calculating frames per second (FPS) in real-time
video processing applications using OpenCV.
"""
from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    """
    Calculate frames per second (FPS) using a moving average.
    
    This class tracks frame times and calculates FPS based on a rolling
    average of the most recent frame durations.
    
    Attributes:
        _start_tick (int): The starting tick count from OpenCV
        _freq (float): Tick frequency conversion factor to milliseconds
        _difftimes (deque): Buffer of recent frame time differences
    """
    def __init__(self, buffer_len=1):
        """
        Initialize the FPS calculator.
        
        Args:
            buffer_len (int, optional): Number of frames to average for FPS calculation.
                Larger values provide smoother but less responsive FPS measurements.
                Defaults to 1.
        """
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        """
        Calculate and return the current FPS.
        
        This method should be called once per frame to update the FPS calculation.
        
        Returns:
            float: The calculated frames per second, rounded to 2 decimal places.
        """
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
