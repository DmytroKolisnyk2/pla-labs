import math
import numpy as np
import cv2 as cv


class Transformations2D:
    @staticmethod
    def rotate(object2, degrees):
        radians = math.radians(degrees)
        rotation_matrix = np.array(
            [
                [math.cos(radians), -math.sin(radians)],
                [math.sin(radians), math.cos(radians)],
            ]
        )
        return np.dot(object2, rotation_matrix.T)

    @staticmethod
    def rotate_opencv_object(object2, angle):
        rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_object = cv.transform(object2.reshape(-1, 1, 2), rotation_matrix)
        return rotated_object.squeeze()

    @staticmethod
    def scale(object2, scale_factor):
        scaling_matrix = np.array([[scale_factor, 0], [0, scale_factor]])
        return np.dot(object2, scaling_matrix.T)

    @staticmethod
    def scale_opencv_object(object2, scale_factor):
        scaling_matrix = np.array([[scale_factor, 0], [0, scale_factor]])
        scaled_object = cv.transform(object2.reshape(-1, 1, 2), scaling_matrix)
        return scaled_object.squeeze()

    @staticmethod
    def mirror(object2, axis):
        if axis == "y":
            mirroring_matrix = np.array([[1, 0], [0, -1]])
        elif axis == "x":
            mirroring_matrix = np.array([[-1, 0], [0, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        return np.dot(object2, mirroring_matrix.T)

    @staticmethod
    def mirror_opencv_object(object2, axis):
        if axis == "y":
            mirroring_matrix = np.array([[1, 0], [0, -1]], dtype=np.float32)
        elif axis == "x":
            mirroring_matrix = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        return cv.transform(object2.reshape(-1, 1, 2), mirroring_matrix).squeeze()

    @staticmethod
    def shear(object2, axis, shear_factor):
        if axis == "x":
            shear_matrix = np.array([[1, shear_factor], [0, 1]])
        elif axis == "y":
            shear_matrix = np.array([[1, 0], [shear_factor, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        return np.dot(object2, shear_matrix.T)

    @staticmethod
    def shear_opencv_object(object2, axis, shear_factor):
        if axis == "x":
            shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        elif axis == "y":
            shear_matrix = np.array([[1, 0, 0], [shear_factor, 1, 0]], dtype=np.float32)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        return cv.transform(object2.reshape(-1, 1, 2), shear_matrix).squeeze()

    @staticmethod
    def universal(object2, matrix):
        return np.dot(object2, matrix)
