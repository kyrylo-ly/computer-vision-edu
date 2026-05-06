"""Geometric transformations for face image correction.

Implements scaling, rotation, and perspective transforms as required by
Laboratory Work 4, Week 7. All operations return a new array and never
modify the input in-place so they are safe to chain freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ScaleResult:
    """Outcome of a scaling operation."""

    image: np.ndarray
    original_size: tuple[int, int]   # (width, height)
    new_size: tuple[int, int]        # (width, height)
    scale_x: float
    scale_y: float
    interpolation: str


@dataclass
class RotationResult:
    """Outcome of a rotation operation."""

    image: np.ndarray
    angle_deg: float
    center: tuple[int, int]
    scale: float


@dataclass
class PerspectiveResult:
    """Outcome of a perspective warp."""

    image: np.ndarray
    transform_matrix: np.ndarray
    src_points: np.ndarray
    dst_points: np.ndarray


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GeometricTransformer:
    """Static helpers for scaling, rotating, and warping images.

    These transforms are commonly applied as a pre-processing step to
    normalise face images before recognition: scaling standardises the
    input resolution, rotation corrects head tilt, and perspective warp
    compensates for camera angle or pose variation.
    """

    # Map human-readable names to OpenCV interpolation constants
    _INTERPOLATIONS: dict[str, int] = {
        "nearest":  cv2.INTER_NEAREST,
        "linear":   cv2.INTER_LINEAR,
        "cubic":    cv2.INTER_CUBIC,
        "area":     cv2.INTER_AREA,
        "lanczos":  cv2.INTER_LANCZOS4,
    }

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    @classmethod
    def scale_by_factor(
        cls,
        image: np.ndarray,
        fx: float,
        fy: float,
        interpolation: str = "linear",
    ) -> ScaleResult:
        """Resize an image by independent X and Y scale factors.

        When shrinking (factor < 1) ``INTER_AREA`` is recommended because it
        computes a proper box-filter average over the removed pixels.  When
        enlarging, ``INTER_CUBIC`` or ``INTER_LINEAR`` produce smooth results.

        Parameters
        ----------
        image:
            Source image in any channel format.
        fx, fy:
            Horizontal and vertical scale factors.  Values > 1 enlarge;
            values < 1 shrink.
        interpolation:
            One of ``"nearest"``, ``"linear"``, ``"cubic"``, ``"area"``,
            ``"lanczos"``.
        """
        if interpolation not in cls._INTERPOLATIONS:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Choose from {sorted(cls._INTERPOLATIONS)}"
            )
        h, w = image.shape[:2]
        interp_flag = cls._INTERPOLATIONS[interpolation]
        resized = cv2.resize(image, None, fx=fx, fy=fy, interpolation=interp_flag)
        new_h, new_w = resized.shape[:2]
        return ScaleResult(
            image=resized,
            original_size=(w, h),
            new_size=(new_w, new_h),
            scale_x=fx,
            scale_y=fy,
            interpolation=interpolation,
        )

    @classmethod
    def scale_to_size(
        cls,
        image: np.ndarray,
        width: int,
        height: int,
        interpolation: str = "linear",
    ) -> ScaleResult:
        """Resize an image to an absolute pixel size.

        Parameters
        ----------
        width, height:
            Target dimensions in pixels.
        """
        if interpolation not in cls._INTERPOLATIONS:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Choose from {sorted(cls._INTERPOLATIONS)}"
            )
        h, w = image.shape[:2]
        interp_flag = cls._INTERPOLATIONS[interpolation]
        resized = cv2.resize(image, (width, height), interpolation=interp_flag)
        return ScaleResult(
            image=resized,
            original_size=(w, h),
            new_size=(width, height),
            scale_x=width / max(w, 1),
            scale_y=height / max(h, 1),
            interpolation=interpolation,
        )

    @classmethod
    def compare_interpolations(
        cls,
        image: np.ndarray,
        fx: float = 0.5,
        fy: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Resize an image with every available interpolation method.

        Returns a dictionary mapping method name to the resized image.
        Useful for qualitative comparison of interpolation artefacts.
        """
        return {
            name: cls.scale_by_factor(image, fx, fy, interpolation=name).image
            for name in cls._INTERPOLATIONS
        }

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    @classmethod
    def rotate(
        cls,
        image: np.ndarray,
        angle_deg: float,
        center: Optional[tuple[int, int]] = None,
        scale: float = 1.0,
        expand: bool = False,
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> RotationResult:
        """Rotate an image around a given centre point.

        The rotation is implemented as an affine warp built from
        ``cv2.getRotationMatrix2D``.  The matrix encodes both rotation and an
        optional uniform scale correction so that it can simultaneously
        compensate for head-tilt and normalise face size.

        Parameters
        ----------
        angle_deg:
            Counter-clockwise rotation in degrees.
        center:
            Pixel coordinate to rotate around.  Defaults to the image centre.
        scale:
            Uniform scale applied jointly with the rotation (1.0 = no change).
        expand:
            When *True* the output canvas is enlarged so the rotated content
            fits without clipping.  When *False* the canvas keeps the original
            size and corners are cropped.
        border_value:
            BGR fill colour for pixels outside the original frame.
        """
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle_deg, scale)

        if expand:
            # Compute the new bounding box after rotation
            cos_a = abs(M[0, 0])
            sin_a = abs(M[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            # Adjust the translation column of the matrix
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            out_size = (new_w, new_h)
        else:
            out_size = (w, h)

        rotated = cv2.warpAffine(
            image, M, out_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return RotationResult(
            image=rotated,
            angle_deg=angle_deg,
            center=center,
            scale=scale,
        )

    @classmethod
    def rotate_steps(
        cls,
        image: np.ndarray,
        angles: Sequence[float] = (-30, -15, 0, 15, 30),
    ) -> dict[str, np.ndarray]:
        """Rotate an image at several angles for comparison.

        Returns a mapping from label string to rotated image.
        """
        return {
            f"{a:+.0f}°": cls.rotate(image, a, expand=True).image
            for a in angles
        }

    # ------------------------------------------------------------------
    # Perspective transformation
    # ------------------------------------------------------------------

    @classmethod
    def perspective_warp(
        cls,
        image: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Optional[tuple[int, int]] = None,
    ) -> PerspectiveResult:
        """Apply a projective (perspective) warp to an image.

        The transform is computed from four point correspondences using
        ``cv2.getPerspectiveTransform``.  It maps any quadrilateral (the source
        region, e.g. a tilted face bounding box) to a rectangle, effectively
        correcting perspective distortion.

        Parameters
        ----------
        src_points:
            Four source points, shape ``(4, 2)``, float32.
        dst_points:
            Four destination points, shape ``(4, 2)``, float32.
        output_size:
            ``(width, height)`` of the output canvas.  Defaults to the input
            image size.
        """
        h, w = image.shape[:2]
        if output_size is None:
            output_size = (w, h)

        src = np.asarray(src_points, dtype=np.float32).reshape(4, 2)
        dst = np.asarray(dst_points, dtype=np.float32).reshape(4, 2)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, output_size)
        return PerspectiveResult(
            image=warped,
            transform_matrix=M,
            src_points=src,
            dst_points=dst,
        )

    @classmethod
    def correct_face_perspective(
        cls,
        image: np.ndarray,
        face_box: tuple[int, int, int, int],
        margin: float = 0.1,
    ) -> PerspectiveResult:
        """Warp a detected face region to a frontal canonical view.

        The four corners of the face bounding box (with an optional margin)
        are mapped to a rectangle whose aspect ratio matches the original box.
        This is a simplified front-normalisation that removes mild in-plane
        perspective shift — sufficient for a pre-processing demonstration.

        Parameters
        ----------
        face_box:
            ``(x, y, w, h)`` in pixels, as returned by Haar cascade detection.
        margin:
            Fractional padding added around the face box (0.1 = 10 %).
        """
        x, y, w, h = face_box
        img_h, img_w = image.shape[:2]

        # Add margin, clamped to image bounds
        mx = int(w * margin)
        my = int(h * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(img_w, x + w + mx)
        y1 = min(img_h, y + h + my)

        src_points = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )
        out_w = x1 - x0
        out_h = y1 - y0
        dst_points = np.array(
            [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32
        )
        return cls.perspective_warp(
            image, src_points, dst_points, output_size=(out_w, out_h)
        )

    @classmethod
    def demo_perspective(
        cls,
        image: np.ndarray,
        skew_x: int = 60,
    ) -> tuple[PerspectiveResult, PerspectiveResult]:
        """Demonstrate perspective distortion and its correction.

        *Forward warp*: applies a keystone distortion that simulates a
        camera looking at the subject from an angle.
        *Inverse warp*: undoes that distortion, recovering the original view.

        Returns a ``(distorted, corrected)`` pair of ``PerspectiveResult``.
        """
        h, w = image.shape[:2]

        # Original rectangle corners
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Keystone: push the top edge inward to simulate perspective
        distorted_dst = np.array(
            [[skew_x, 0], [w - skew_x, 0], [w, h], [0, h]], dtype=np.float32
        )

        forward = cls.perspective_warp(image, src, distorted_dst)

        # Correction: map the distorted corners back to the rectangle
        corrected = cls.perspective_warp(
            forward.image, distorted_dst, src, output_size=(w, h)
        )
        return forward, corrected
