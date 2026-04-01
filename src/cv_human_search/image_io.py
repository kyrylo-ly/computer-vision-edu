"""Image loading, diagnostics, and display helpers.

The loader relies on Pillow for format robustness and converts the image into
OpenCV's BGR convention so the rest of the pipeline can use cv2 operations
without repeatedly swapping channel order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageMetadata:
    """Basic diagnostics for an image array."""

    path: str
    width: int
    height: int
    channels: int
    dtype: str
    mode: str


class ImageLoader:
    """Load common image formats and expose diagnostic helpers."""

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Load an image robustly using Pillow and return a BGR array.

        Pillow handles format decoding for JPG, PNG, and BMP reliably. The
        image is normalized into 8-bit RGB first and then converted to BGR so
        OpenCV processing can proceed naturally.
        """

        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image_path.suffix.lower() not in ImageLoader.valid_extensions:
            raise ValueError(
                f"Unsupported extension '{image_path.suffix}'. Supported: "
                f"{sorted(ImageLoader.valid_extensions)}"
            )

        with Image.open(image_path) as pil_image:
            pil_image = pil_image.convert("RGB")
            rgb_image = np.asarray(pil_image, dtype=np.uint8)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image

    @staticmethod
    def load_gray_image(path: str | Path) -> np.ndarray:
        """Load an image and convert it directly to grayscale."""

        image = ImageLoader.load_image(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_metadata(image: np.ndarray, path: str | Path | None = None) -> ImageMetadata:
        """Return a metadata summary for diagnostics and logging."""

        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        image_path = str(path) if path is not None else "<memory>"
        mode = "GRAY" if image.ndim == 2 else "BGR"
        return ImageMetadata(
            path=image_path,
            width=width,
            height=height,
            channels=channels,
            dtype=str(image.dtype),
            mode=mode,
        )

    @staticmethod
    def print_metadata(image: np.ndarray, path: str | Path | None = None) -> None:
        """Print image metadata in a human-readable format."""

        metadata = ImageLoader.get_metadata(image, path)
        for key, value in asdict(metadata).items():
            print(f"{key}: {value}")

    @staticmethod
    def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR image into RGB for plotting libraries."""

        if image_bgr.ndim == 2:
            return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_bgr(image_rgb: np.ndarray) -> np.ndarray:
        """Convert an RGB image into BGR for OpenCV operations."""

        if image_rgb.ndim == 2:
            return image_rgb
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def ensure_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize an image to uint8 if needed.

        This is useful when intermediate steps create float arrays. The values
        are clipped into the displayable range before conversion.
        """

        if image.dtype == np.uint8:
            return image
        clipped = np.clip(image, 0, 255)
        return clipped.astype(np.uint8)


