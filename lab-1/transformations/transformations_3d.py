import math
import numpy as np


class Transformations3D:
    @staticmethod
    def rotate(object3, axis, degrees):
        radians = math.radians(degrees)
        if axis == "x":
            rotation_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(radians), -math.sin(radians)],
                    [0, math.sin(radians), math.cos(radians)],
                ]
            )
        elif axis == "y":
            rotation_matrix = np.array(
                [
                    [math.cos(radians), 0, math.sin(radians)],
                    [0, 1, 0],
                    [-math.sin(radians), 0, math.cos(radians)],
                ]
            )
        elif axis == "z":
            rotation_matrix = np.array(
                [
                    [math.cos(radians), -math.sin(radians), 0],
                    [math.sin(radians), math.cos(radians), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        return np.dot(object3, rotation_matrix.T)

    @staticmethod
    def scale(object3, scale_factor):
        scaling_matrix = np.array(
            [
                [scale_factor, 0, 0],
                [0, scale_factor, 0],
                [0, 0, scale_factor],
            ]
        )
        return np.dot(object3, scaling_matrix.T)
