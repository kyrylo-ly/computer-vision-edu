"""Feature extraction and human detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FeatureSummary:
    """Lightweight report of feature extraction results."""

    method: str
    num_keypoints: int = 0
    num_detections: int = 0
    descriptor_shape: Optional[tuple[int, ...]] = None


class FeatureExtractor:
    """Classical feature detectors and descriptors."""

    @staticmethod
    def detect_contours(binary_image: np.ndarray,
                        min_area: float = 100.0) -> List[np.ndarray]:
        """Detect contours from a binary mask.

        Contours are extracted using border following on connected components.
        Filtering by area removes tiny fragments that are usually not useful in
        face or human localization.
        """

        if binary_image.ndim != 2:
            raise ValueError("Contour detection expects a binary or grayscale image.")

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def sift_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray], np.ndarray]:
        """Detect SIFT keypoints and descriptors.

        SIFT finds scale-space extrema in Difference-of-Gaussians space and
        computes rotation-invariant descriptors based on local gradient
        histograms. It is robust but may require opencv-contrib builds.
        """

        gray = FeatureExtractor._gray(image)
        if not hasattr(cv2, "SIFT_create"):
            return [], None, image.copy()

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        annotated = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return keypoints, descriptors, annotated

    @staticmethod
    def orb_features(image: np.ndarray, max_features: int = 1000) -> tuple[list[cv2.KeyPoint], Optional[np.ndarray], np.ndarray]:
        """Detect ORB keypoints and descriptors.

        ORB combines oriented FAST corners with rotated BRIEF descriptors.
        Compared with SIFT, it is faster and free of patent restrictions.
        """

        gray = FeatureExtractor._gray(image)
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        annotated = cv2.drawKeypoints(
            image, keypoints, None, color=(0, 255, 0)
        )
        return keypoints, descriptors, annotated

    @staticmethod
    def hog_human_detection(image: np.ndarray) -> tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """Detect humans using the classical HOG + linear SVM pipeline.

        HOG aggregates edge orientations into local cell histograms and then
        normalizes them over blocks. The resulting descriptor is fed into a
        pretrained linear SVM that responds strongly to pedestrian-like shapes.
        """

        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = detector.detectMultiScale(
            image, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        annotated = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return list(boxes), annotated

    @staticmethod
    def compare_methods(image: np.ndarray) -> dict[str, Any]:
        """Run SIFT, ORB, and HOG on the same input and summarize results."""

        sift_keypoints, sift_descriptors, sift_vis = FeatureExtractor.sift_features(image)
        orb_keypoints, orb_descriptors, orb_vis = FeatureExtractor.orb_features(image)
        hog_boxes, hog_vis = FeatureExtractor.hog_human_detection(image)

        return {
            "sift": {
                "summary": FeatureSummary(
                    method="SIFT",
                    num_keypoints=len(sift_keypoints),
                    descriptor_shape=None if sift_descriptors is None else sift_descriptors.shape,
                ),
                "visualization": sift_vis,
            },
            "orb": {
                "summary": FeatureSummary(
                    method="ORB",
                    num_keypoints=len(orb_keypoints),
                    descriptor_shape=None if orb_descriptors is None else orb_descriptors.shape,
                ),
                "visualization": orb_vis,
            },
            "hog": {
                "summary": FeatureSummary(
                    method="HOG",
                    num_detections=len(hog_boxes),
                ),
                "visualization": hog_vis,
            },
        }


class HumanDetector:
    """Convenience wrapper for contour and human detection stages."""

    @staticmethod
    def face_like_contours(binary_image: np.ndarray, min_area: float = 500.0) -> List[np.ndarray]:
        """Return contours that may correspond to face regions.

        This stage is intentionally classical and geometry-based. It is most
        useful after segmentation has produced a clean foreground mask.
        """

        return FeatureExtractor.detect_contours(binary_image, min_area=min_area)

    @staticmethod
    def human_boxes(image: np.ndarray) -> tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """Return HOG-based human detections."""

        return FeatureExtractor.hog_human_detection(image)

