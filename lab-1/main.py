import numpy as np
import cv2 as cv
from plotting import Plotting
from transformations import Transformations2D, Transformations3D, ImageTransformations


def matrix_plotting():
    # 2D transformation examples with a star shape
    object2_array = np.array(
        [
            [0, 1],
            [0.2, 0.2],
            [1, 0.2],
            [0.4, -0.1],
            [0.6, -1],
            [0, -0.5],
            [-0.6, -1],
            [-0.4, -0.1],
            [-1, 0.2],
            [-0.2, 0.2],
            [0, 1],
        ]
    )
    original_object = object2_array.copy()

    # Example 1: Rotation
    by_cv = Transformations2D.rotate_opencv_object(object2_array, 45)
    Plotting.plot_2d(original_object, by_cv, "Rotation by OpenCV")

    by_custom = Transformations2D.rotate(object2_array, 45)
    Plotting.plot_2d(original_object, by_custom, "Rotation by Custom Function")

    # Example 2: Scaling
    by_cv = Transformations2D.scale_opencv_object(object2_array, 2)
    Plotting.plot_2d(original_object, by_cv, "Scaling by OpenCV")

    by_custom = Transformations2D.scale(object2_array, 2)
    Plotting.plot_2d(original_object, by_custom, "Scaling by Custom Function")

    # Example 3: Mirroring
    by_cv = Transformations2D.mirror_opencv_object(object2_array, "y")
    Plotting.plot_2d(original_object, by_cv, "Mirroring by OpenCV")

    by_custom = Transformations2D.mirror(object2_array, "y")
    Plotting.plot_2d(original_object, by_custom, "Mirroring by Custom Function")

    # Example 4: Shear
    by_cv = Transformations2D.shear_opencv_object(object2_array, "x", 0.5)
    Plotting.plot_2d(original_object, by_cv, "Shear by OpenCV")

    by_custom = Transformations2D.shear(object2_array, "x", 0.5)
    Plotting.plot_2d(original_object, by_custom, "Shear by Custom Function")

    # Example 5: Custom Function
    matrix = np.array([[0, -5], [10, -3]])
    by_custom = Transformations2D.universal(object2_array, matrix)

    Plotting.plot_2d(original_object, by_custom, "Custom Function")

    # 3D transformation examples with an asymmetrical diamond shape
    object3_array = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [-0.5, 1, 0], [-1, 0, 0], [0, -1, 0]]
    )

    object3 = Transformations3D.rotate(object3_array, "x", 45)
    Plotting.plot_3d(object3, "Rotation by x-axis")

    object3 = Transformations3D.rotate(object3, "y", 45)
    Plotting.plot_3d(object3, "Rotation by y-axis")

    object3 = Transformations3D.rotate(object3, "z", 45)
    Plotting.plot_3d(object3, "Rotation by z-axis")

    object3 = Transformations3D.scale(object3, 5)
    Plotting.plot_3d(object3, "Scaling")


def image_plotting():
    # 2D transformation on image
    img_path = "./assets/image.jpg"
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    if img is None:
        print("Error: Could not read the image.")
    else:
        Plotting.plot_image("Original Image", img)

        # Scaling the image
        dst = ImageTransformations.scale_image(img, 2)
        Plotting.plot_image("Scaled Image", dst)

        # Rotating the image
        dst = ImageTransformations.rotate_image(img, 45)
        Plotting.plot_image("Rotated Image", dst)

        # Mirroring the image
        dst = ImageTransformations.mirror_image(img, "y")
        Plotting.plot_image("Mirrored Image", dst)

        # Scale, rotate, and mirror the image
        dst = ImageTransformations.rotate_image(img, 45)
        dst = ImageTransformations.scale_image(dst, 0.5)
        dst = ImageTransformations.mirror_image(dst, "x")
        Plotting.plot_image("Custom Image", dst)

        # Scale, rotate, and mirror the image in other order
        dst = ImageTransformations.scale_image(img, 0.5)
        dst = ImageTransformations.mirror_image(dst, "x")
        dst = ImageTransformations.rotate_image(dst, 45)
        Plotting.plot_image("Custom Image in other order", dst)


def main():
    # matrix_plotting()
    image_plotting()


main()
