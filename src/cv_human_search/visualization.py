"""Visualization helpers for OpenCV and Matplotlib."""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .image_io import ImageLoader


class Visualizer:
    """Collection of display and plotting helpers."""

    @staticmethod
    def show_opencv(window_name: str, image: np.ndarray, wait_ms: int = 1) -> None:
        """Display an image using OpenCV's native GUI window."""

        cv2.imshow(window_name, ImageLoader.ensure_uint8(image))
        cv2.waitKey(wait_ms)

    @staticmethod
    def show_matplotlib(image: np.ndarray, title: str = "Image") -> None:
        """Display an image using Matplotlib in RGB order."""

        plt.figure(figsize=(8, 6))
        plt.title(title)
        rgb_image = ImageLoader.to_rgb(image)
        plt.imshow(rgb_image, cmap="gray" if image.ndim == 2 else None)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_image_grid(images: Sequence[np.ndarray], titles: Sequence[str],
                        cols: int = 2, figsize: tuple[int, int] = (14, 8)) -> None:
        """Plot multiple images in a grid for qualitative comparison."""

        if len(images) != len(titles):
            raise ValueError("The number of images must match the number of titles.")

        rows = int(np.ceil(len(images) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for axis, image, title in zip(axes, images, titles):
            axis.imshow(ImageLoader.to_rgb(image), cmap="gray" if image.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")

        for axis in axes[len(images):]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_histogram(gray_image: np.ndarray, title: str = "Brightness Histogram") -> None:
        """Plot the brightness distribution of a grayscale image."""

        if gray_image.ndim != 2:
            raise ValueError("Histogram plotting expects a grayscale image.")

        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel()
        plt.figure(figsize=(8, 4))
        plt.plot(histogram, color="black")
        plt.title(title)
        plt.xlabel("Intensity")
        plt.ylabel("Pixel count")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: Iterable[tuple[int, int, int, int]],
                   color: tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """Draw rectangular detections on a copy of the input image."""

        output = image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        return output

    @staticmethod
    def draw_contours(image: np.ndarray, contours: Sequence[np.ndarray],
                      color: tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """Overlay contours on a copy of the image."""

        output = image.copy()
        cv2.drawContours(output, list(contours), -1, color, thickness)
        return output

