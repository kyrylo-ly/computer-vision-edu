"""Brightness analysis and contrast enhancement utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HistogramResult:
    """Brightness histogram data."""

    histogram: np.ndarray
    bins: np.ndarray


class ContrastEnhancer:
    """Apply global and local contrast enhancement methods."""

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """Convert a BGR or RGB image to grayscale."""

        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def compute_brightness_histogram(image: np.ndarray) -> HistogramResult:
        """Compute a grayscale brightness histogram.

        The histogram approximates the discrete probability distribution of
        pixel intensities. Dark or low-contrast images tend to have a narrow
        histogram band, while well-exposed images occupy more of the range.
        """

        gray = ContrastEnhancer.to_gray(image)
        histogram, bins = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        return HistogramResult(histogram=histogram, bins=bins)

    @staticmethod
    def equalize_global_histogram(image: np.ndarray) -> np.ndarray:
        """Enhance contrast using global histogram equalization.

        For color images the operation is applied to the luminance channel in
        YCrCb space to avoid shifting chromaticity. The cumulative distribution
        function is redistributed across the full intensity range, which tends
        to stretch clustered tonal values.
        """

        if image.ndim == 2:
            return cv2.equalizeHist(image)

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_equalized = cv2.equalizeHist(y_channel)
        merged = cv2.merge((y_equalized, cr_channel, cb_channel))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def equalize_clahe(image: np.ndarray,
                       clip_limit: float = 2.0,
                       tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        """Enhance local contrast using CLAHE.

        CLAHE divides the image into tiles, equalizes each tile, and clips
        histograms above a threshold before redistribution. The clipping step
        prevents noise spikes from dominating the output, which is particularly
        useful for faces where fine texture should be preserved.
        """

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if image.ndim == 2:
            return clahe.apply(image)

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_equalized = clahe.apply(y_channel)
        merged = cv2.merge((y_equalized, cr_channel, cb_channel))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


compute_brightness_histogram = ContrastEnhancer.compute_brightness_histogram
