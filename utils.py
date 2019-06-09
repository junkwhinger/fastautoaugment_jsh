# Reference
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

import logging
import json


class Params():
    """Class that holds a hyper-parameter set for an experiment"""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def set_logger(log_path):
    """Method that set the logger ready for an experiment"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.data_points = 0
        self.total_loss = 0
        self.total_corrects = 0

        self.avg_loss = 0.0
        self.accuracy = 0.0
        self.error = 0.0

    def update(self, loss, corrects, data_points):
        self.data_points += data_points
        self.total_loss += loss
        self.total_corrects += corrects

    def calculate(self):
        self.avg_loss = self.total_loss / float(self.data_points)
        self.accuracy = float(self.total_corrects) / float(self.data_points)
        self.error = 1.0 - self.accuracy

    def __call__(self):
        return self.avg_loss, self.accuracy, self.error, self.data_points
