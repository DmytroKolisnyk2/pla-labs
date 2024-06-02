import cv2 as cv

class ImageTransformations:
    @staticmethod
    def rotate_image(img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        return cv.warpAffine(img, rotation_matrix, (w, h))

    @staticmethod
    def scale_image(img, scale_factor):
        return cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)

    @staticmethod
    def mirror_image(img, axis):
        if axis == "x":
            return cv.flip(img, 0)
        elif axis == "y":
            return cv.flip(img, 1)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
