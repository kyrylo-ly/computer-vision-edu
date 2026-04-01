"""Noise reduction and sharpening operations."""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocessing filters used before segmentation and feature extraction."""

    @staticmethod
    def gaussian_denoise(image: np.ndarray, kernel_size: tuple[int, int] = (5, 5),
                         sigma: float = 0.0) -> np.ndarray:
        """Apply Gaussian smoothing.

        Gaussian filtering is a low-pass operation that weights nearby pixels
        according to a normal distribution. It reduces sensor noise while
        preserving broad facial structure better than a simple box filter.
        """

        return cv2.GaussianBlur(image, kernel_size, sigma)

    @staticmethod
    def median_denoise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filtering.

        The median is robust to impulse noise because the output is the middle
        value from the local neighborhood rather than the arithmetic mean.
        This is useful when isolated bright or dark outliers appear in the
        background or on a face image.
        """

        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def bilateral_denoise(image: np.ndarray, diameter: int = 9,
                          sigma_color: float = 75.0,
                          sigma_space: float = 75.0) -> np.ndarray:
        """Apply bilateral filtering.

        The bilateral filter combines a spatial Gaussian with an intensity
        similarity Gaussian. Pixels are averaged only when they are nearby and
        similar in color, so edges remain sharp. This makes it the preferred
        denoiser when preserving facial boundaries is important.
        """

        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    @staticmethod
    def sharpen_laplacian(image: np.ndarray) -> np.ndarray:
        """Sharpen an image using the Laplacian operator.

        The Laplacian estimates the second spatial derivative. Subtracting a
        scaled version of that response from the original image accentuates
        intensity transitions, which makes edges more visible.
        """

        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image.astype(np.float64) - laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: tuple[int, int] = (0, 0),
                     sigma: float = 1.5, amount: float = 1.2) -> np.ndarray:
        """Apply unsharp masking.

        The image is blurred, the blur is subtracted from the original to
        obtain a high-frequency detail layer, and the detail layer is then
        added back with a gain factor.
        """

        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    @classmethod
    def full_preprocess(cls, image: np.ndarray, denoise_method: str = "bilateral",
                        sharpen_method: str = "unsharp") -> np.ndarray:
        """Run a configurable denoise-plus-sharpen pipeline."""

        denoise_map = {
            "gaussian": cls.gaussian_denoise,
            "median": cls.median_denoise,
            "bilateral": cls.bilateral_denoise,
        }
        sharpen_map = {
            "laplacian": cls.sharpen_laplacian,
            "unsharp": cls.unsharp_mask,
        }

        if denoise_method not in denoise_map:
            raise ValueError(f"Unknown denoise method: {denoise_method}")
        if sharpen_method not in sharpen_map:
            raise ValueError(f"Unknown sharpen method: {sharpen_method}")

        denoised = denoise_map[denoise_method](image)
        sharpened = sharpen_map[sharpen_method](denoised)
        return sharpened

