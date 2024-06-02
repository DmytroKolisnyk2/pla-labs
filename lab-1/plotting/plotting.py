import matplotlib.pyplot as plt
import cv2 as cv


SHOW_TIME_SEC = 3


class Plotting:
    @staticmethod
    def plot_2d(original_object, transformed_object, title):
        plt.plot(
            original_object[:, 0], original_object[:, 1], "bo-", label="Original Object"
        )
        plt.plot(
            transformed_object[:, 0],
            transformed_object[:, 1],
            "ro-",
            label="Transformed Object",
        )
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.title(title)
        plt.axis("equal")
        plt.legend()

        plt.show(block=False)
        plt.pause(SHOW_TIME_SEC)
        plt.close()

    @staticmethod
    def plot_3d(object3, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(object3)):
            for j in range(i, len(object3)):
                ax.plot3D(*zip(object3[i], object3[j]), color="blue")

        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title(title)

        plt.show(block=False)
        plt.pause(SHOW_TIME_SEC)
        plt.close()

    @staticmethod
    def plot_image(title, dst):
        cv.imshow(title, dst)
        cv.waitKey(SHOW_TIME_SEC * 1000)
        cv.destroyAllWindows()
