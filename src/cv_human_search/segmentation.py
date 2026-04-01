"""Image segmentation routines for isolating faces and foreground objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Container for segmentation outputs."""

    mask: np.ndarray
    segmented_image: np.ndarray


class Segmenter:
    """Segmentation algorithms used for foreground separation."""

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale for thresholding steps."""

        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def otsu_binarize(image: np.ndarray) -> np.ndarray:
        """Perform Otsu's adaptive global thresholding.

        Otsu's method chooses the threshold that minimizes the weighted within-
        class variance. In effect, it assumes a bimodal histogram and searches
        for the value that best separates foreground from background.
        """

        gray = Segmenter.to_gray(image)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def grabcut_segment(image: np.ndarray,
                        rect: Optional[tuple[int, int, int, int]] = None,
                        iterations: int = 5) -> SegmentationResult:
        """Extract a precise foreground mask using GrabCut.

        GrabCut models foreground and background with Gaussian Mixture Models.
        Starting from a rectangle or an initialized mask, it repeatedly updates
        the pixel labels and boundary estimates until convergence.
        """

        mask = np.zeros(image.shape[:2], np.uint8)
        if rect is None:
            height, width = image.shape[:2]
            margin_x = int(width * 0.1)
            margin_y = int(height * 0.1)
            rect = (
                margin_x,
                margin_y,
                max(1, width - 2 * margin_x),
                max(1, height - 2 * margin_y),
            )

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bg_model, fg_model, iterations,
                    cv2.GC_INIT_WITH_RECT)

        output_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype("uint8")
        segmented = image * output_mask[:, :, np.newaxis]
        return SegmentationResult(mask=output_mask * 255, segmented_image=segmented)

    @staticmethod
    def watershed_segment(image: np.ndarray) -> SegmentationResult:
        """Separate overlapping objects using the watershed transform.

        The method interprets the image as a topographic surface. After
        generating sure foreground and sure background markers, the watershed
        flood fills the landscape from marker seeds and builds ridges at object
        boundaries. These ridges are especially useful when objects overlap.
        """

        gray = Segmenter.to_gray(image)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 0.4 * dist_transform.max(), 255, 0
        )
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_input = image.copy()
        if watershed_input.ndim == 2:
            watershed_input = cv2.cvtColor(watershed_input, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_input, markers)

        segmented = np.zeros_like(watershed_input)
        segmented[markers > 1] = watershed_input[markers > 1]
        return SegmentationResult(mask=(markers > 1).astype(np.uint8) * 255,
                                  segmented_image=segmented)

