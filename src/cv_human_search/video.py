"""Real-time video analysis with background subtraction and optical flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    """Summary of motion between two frames."""

    mean_dx: float = 0.0
    mean_dy: float = 0.0
    velocity: float = 0.0
    tracked_points: int = 0


class VideoAnalyzer:
    """Process a video stream for moving subject isolation and tracking."""

    def __init__(self, source: int | str = 0, bg_method: str = "MOG2",
                 flow_method: str = "lk", display: bool = True) -> None:
        self.source = source
        self.bg_method = bg_method.lower()
        self.flow_method = flow_method.lower()
        self.display = display
        self.capture: Optional[cv2.VideoCapture] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.bg_subtractor = self._create_bg_subtractor(self.bg_method)

    @staticmethod
    def _create_bg_subtractor(method: str):
        if method == "knn":
            return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0)
        return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    @staticmethod
    def _clean_foreground(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned

    @staticmethod
    def _estimate_motion(prev_gray: np.ndarray, gray: np.ndarray,
                         method: str = "lk",
                         prev_points: Optional[np.ndarray] = None) -> tuple[np.ndarray, MotionEstimate, Optional[np.ndarray]]:
        """Estimate frame-to-frame motion using optical flow."""

        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            motion = MotionEstimate(
                mean_dx=float(np.mean(flow[..., 0])),
                mean_dy=float(np.mean(flow[..., 1])),
                velocity=float(np.mean(magnitude)),
                tracked_points=int(magnitude.size),
            )
            return flow_vis, motion, None

        if prev_points is None or len(prev_points) < 20:
            prev_points = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
            )
            if prev_points is None:
                motion = MotionEstimate()
                return canvas, motion, None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        if next_points is None or status is None:
            motion = MotionEstimate()
            return canvas, motion, prev_points

        good_new = next_points[status.flatten() == 1]
        good_old = prev_points[status.flatten() == 1]

        if len(good_new) == 0:
            motion = MotionEstimate()
            return canvas, motion, None

        displacements = good_new - good_old
        mean_dx = float(np.mean(displacements[:, 0]))
        mean_dy = float(np.mean(displacements[:, 1]))
        velocity = float(np.mean(np.linalg.norm(displacements, axis=1)))

        for new_point, old_point in zip(good_new, good_old):
            x_new, y_new = new_point.ravel().astype(int)
            x_old, y_old = old_point.ravel().astype(int)
            cv2.arrowedLine(canvas, (x_old, y_old), (x_new, y_new), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(canvas, (x_new, y_new), 2, (0, 0, 255), -1)

        motion = MotionEstimate(
            mean_dx=mean_dx,
            mean_dy=mean_dy,
            velocity=velocity,
            tracked_points=int(len(good_new)),
        )
        return canvas, motion, good_new.reshape(-1, 1, 2)

    def run(self) -> None:
        """Start the live capture loop."""

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        while True:
            success, frame = self.capture.read()
            if not success:
                break

            frame = cv2.resize(frame, None, fx=0.9, fy=0.9)
            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask = self._clean_foreground(fg_mask)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_vis = frame.copy()
            motion = MotionEstimate()
            if self.prev_gray is not None:
                motion_vis, motion, self.prev_points = self._estimate_motion(
                    self.prev_gray, gray, self.flow_method, self.prev_points
                )

            self.prev_gray = gray

            cv2.putText(
                motion_vis,
                f"dx={motion.mean_dx:.2f} dy={motion.mean_dy:.2f} vel={motion.velocity:.2f} pts={motion.tracked_points}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
            )

            fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, fg_color, motion_vis])
            cv2.imshow("Video Analysis | Frame | Foreground Mask | Motion", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        self.release()

    def release(self) -> None:
        """Release capture resources and close GUI windows."""

        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

