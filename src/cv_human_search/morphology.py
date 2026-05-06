"""Morphological operations for segmentation quality improvement.

Implements erosion, dilation, opening, and closing as required by
Laboratory Work 4, Week 8.  All methods accept both binary masks and
grayscale/colour images and return a new array.

Morphological operations treat the image as a set of pixel positions and
apply a structuring element (kernel) to grow or shrink that set:
  - **Erosion** — removes pixels at object boundaries (shrinks white regions).
  - **Dilation** — adds pixels at boundaries (grows white regions).
  - **Opening** (erosion → dilation) — removes small noise specks while
    keeping large objects intact.
  - **Closing** (dilation → erosion) — fills small holes inside objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


class KernelFactory:
    """Convenience factory for common structuring element shapes."""

    @staticmethod
    def rect(size: int = 3) -> np.ndarray:
        """Rectangular (box) kernel of shape ``size × size``."""
        return cv2.getStructuringElement(
            cv2.MORPH_RECT, (size, size)
        )

    @staticmethod
    def ellipse(size: int = 3) -> np.ndarray:
        """Elliptical kernel — softer boundaries than rectangle."""
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)
        )

    @staticmethod
    def cross(size: int = 3) -> np.ndarray:
        """Cross-shaped kernel — useful for thin elongated structures."""
        return cv2.getStructuringElement(
            cv2.MORPH_CROSS, (size, size)
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MorphResult:
    """Output of a morphological operation together with diagnostic data."""

    operation: str
    image: np.ndarray
    kernel_size: int
    kernel_shape: str
    iterations: int


@dataclass
class MorphComparison:
    """A collection of related morph results for side-by-side display."""

    original: np.ndarray
    results: list[MorphResult] = field(default_factory=list)

    def as_image_list(self) -> list[np.ndarray]:
        return [self.original] + [r.image for r in self.results]

    def as_title_list(self) -> list[str]:
        original_label = "Original"
        labels = [
            f"{r.operation.capitalize()}\nkernel={r.kernel_size}×{r.kernel_size} "
            f"({r.kernel_shape}) ×{r.iterations}"
            for r in self.results
        ]
        return [original_label] + labels


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MorphologyProcessor:
    """Classical morphological operations applied to binary or grey images.

    Each public method accepts an image plus structuring-element parameters
    and returns a ``MorphResult``.  The class also provides higher-level
    comparison helpers that run multiple operations in sequence so the effect
    on segmentation quality can be inspected visually.
    """

    _SHAPES: dict[str, callable] = {
        "rect":    KernelFactory.rect,
        "ellipse": KernelFactory.ellipse,
        "cross":   KernelFactory.cross,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _kernel(cls, size: int, shape: str) -> np.ndarray:
        if shape not in cls._SHAPES:
            raise ValueError(
                f"Unknown kernel shape '{shape}'. "
                f"Choose from {sorted(cls._SHAPES)}"
            )
        return cls._SHAPES[shape](size)

    # ------------------------------------------------------------------
    # Primitive operations
    # ------------------------------------------------------------------

    @classmethod
    def erode(
        cls,
        image: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "rect",
        iterations: int = 1,
    ) -> MorphResult:
        """Apply morphological erosion.

        Erosion shrinks bright (white) regions by replacing each pixel with
        the minimum value in its neighbourhood defined by the kernel.  On
        binary masks this removes isolated foreground pixels and narrows thin
        protrusions — useful for eliminating small noise specks that remain
        after thresholding.

        Parameters
        ----------
        kernel_size:
            Side length of the square structuring element in pixels.
        kernel_shape:
            ``"rect"``, ``"ellipse"``, or ``"cross"``.
        iterations:
            Number of times the operation is repeated.
        """
        k = cls._kernel(kernel_size, kernel_shape)
        eroded = cv2.erode(image, k, iterations=iterations)
        return MorphResult(
            operation="erosion",
            image=eroded,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=iterations,
        )

    @classmethod
    def dilate(
        cls,
        image: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "rect",
        iterations: int = 1,
    ) -> MorphResult:
        """Apply morphological dilation.

        Dilation grows bright regions by replacing each pixel with the maximum
        in its kernel neighbourhood.  On binary masks it closes narrow gaps
        between nearby objects and strengthens the outline of detected regions.

        Parameters
        ----------
        kernel_size:
            Side length of the structuring element.
        kernel_shape:
            ``"rect"``, ``"ellipse"``, or ``"cross"``.
        iterations:
            Number of times the operation is repeated.
        """
        k = cls._kernel(kernel_size, kernel_shape)
        dilated = cv2.dilate(image, k, iterations=iterations)
        return MorphResult(
            operation="dilation",
            image=dilated,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=iterations,
        )

    @classmethod
    def open(
        cls,
        image: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "rect",
        iterations: int = 1,
    ) -> MorphResult:
        """Apply morphological opening (erosion followed by dilation).

        Opening eliminates small bright blobs and smooths the contours of
        larger objects without significantly changing their area.  It is the
        standard post-processing step after Otsu or background-subtraction
        masks to remove pepper noise.

        Parameters
        ----------
        kernel_size:
            Side length of the structuring element.
        kernel_shape:
            ``"rect"``, ``"ellipse"``, or ``"cross"``.
        iterations:
            Iteration count applied to each of the two sub-operations.
        """
        k = cls._kernel(kernel_size, kernel_shape)
        opened = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, k, iterations=iterations
        )
        return MorphResult(
            operation="opening",
            image=opened,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=iterations,
        )

    @classmethod
    def close(
        cls,
        image: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "rect",
        iterations: int = 1,
    ) -> MorphResult:
        """Apply morphological closing (dilation followed by erosion).

        Closing fills small dark holes inside bright objects and bridges
        narrow gaps between nearby regions.  It is complementary to opening:
        while opening removes foreground noise, closing repairs foreground
        breaks left by shadows or occlusions.

        Parameters
        ----------
        kernel_size:
            Side length of the structuring element.
        kernel_shape:
            ``"rect"``, ``"ellipse"``, or ``"cross"``.
        iterations:
            Iteration count applied to each of the two sub-operations.
        """
        k = cls._kernel(kernel_size, kernel_shape)
        closed = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, k, iterations=iterations
        )
        return MorphResult(
            operation="closing",
            image=closed,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Derived operations
    # ------------------------------------------------------------------

    @classmethod
    def gradient(
        cls,
        image: np.ndarray,
        kernel_size: int = 3,
        kernel_shape: str = "rect",
    ) -> MorphResult:
        """Compute the morphological gradient (dilation − erosion).

        The gradient highlights object boundaries by returning the difference
        between the dilated and eroded images.  The result is bright only at
        the edges of foreground regions, making it useful for border detection.
        """
        k = cls._kernel(kernel_size, kernel_shape)
        grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, k)
        return MorphResult(
            operation="gradient",
            image=grad,
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            iterations=1,
        )

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    @classmethod
    def compare_operations(
        cls,
        binary_mask: np.ndarray,
        kernel_size: int = 5,
        kernel_shape: str = "ellipse",
        iterations: int = 2,
    ) -> MorphComparison:
        """Run erosion, dilation, opening, and closing on the same mask.

        Returns a ``MorphComparison`` whose ``as_image_list()`` and
        ``as_title_list()`` methods are ready to pass to ``Visualizer``.

        Parameters
        ----------
        binary_mask:
            Single-channel binary image (uint8, values 0 or 255).
        kernel_size, kernel_shape, iterations:
            Structuring element parameters shared by all operations.
        """
        comparison = MorphComparison(original=binary_mask)
        for op in ("erode", "dilate", "open", "close"):
            method = getattr(cls, op)
            result = method(
                binary_mask,
                kernel_size=kernel_size,
                kernel_shape=kernel_shape,
                iterations=iterations,
            )
            comparison.results.append(result)
        return comparison

    @classmethod
    def compare_kernel_sizes(
        cls,
        binary_mask: np.ndarray,
        operation: str = "open",
        sizes: Sequence[int] = (3, 5, 7, 11),
        kernel_shape: str = "ellipse",
    ) -> MorphComparison:
        """Show the effect of increasing the structuring element size.

        Parameters
        ----------
        binary_mask:
            Input binary mask.
        operation:
            Which operation to sweep: ``"erode"``, ``"dilate"``,
            ``"open"``, or ``"close"``.
        sizes:
            Sequence of kernel sizes to test.
        """
        if operation not in ("erode", "dilate", "open", "close"):
            raise ValueError(
                f"Unknown operation '{operation}'. "
                "Choose from erode, dilate, open, close."
            )
        method = getattr(cls, operation)
        comparison = MorphComparison(original=binary_mask)
        for size in sizes:
            result = method(binary_mask, kernel_size=size, kernel_shape=kernel_shape)
            comparison.results.append(result)
        return comparison

    @classmethod
    def improve_segmentation(
        cls,
        raw_mask: np.ndarray,
        open_kernel: int = 3,
        close_kernel: int = 7,
        kernel_shape: str = "ellipse",
    ) -> tuple[np.ndarray, MorphComparison]:
        """Apply a standard open→close pipeline to clean a segmentation mask.

        The two-stage approach first removes isolated noise pixels (opening)
        and then fills holes in the detected regions (closing).  Returns both
        the final cleaned mask and a ``MorphComparison`` of intermediate steps.

        Parameters
        ----------
        raw_mask:
            Binary mask as produced by Otsu or Watershed segmentation.
        open_kernel:
            Kernel size for the opening step (noise removal).
        close_kernel:
            Kernel size for the closing step (hole filling).
        kernel_shape:
            Structuring element shape used for both steps.
        """
        k_open = cls._kernel(open_kernel, kernel_shape)
        k_close = cls._kernel(close_kernel, kernel_shape)

        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=2)

        comparison = MorphComparison(original=raw_mask)
        comparison.results.append(
            MorphResult("opening",  opened, open_kernel,  kernel_shape, 2)
        )
        comparison.results.append(
            MorphResult("closing",  closed, close_kernel, kernel_shape, 2)
        )
        return closed, comparison
