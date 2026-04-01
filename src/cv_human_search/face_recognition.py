"""Face detection and optional LBPH recognition utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FacePrediction:
    """Recognition output for a single face region."""

    box: tuple[int, int, int, int]
    label: str
    confidence: float


class FaceDetector:
    """Classical Haar-cascade face detector."""

    def __init__(self, cascade_path: Optional[str] = None) -> None:
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_faces(self, image: np.ndarray,
                     scale_factor: float = 1.1,
                     min_neighbors: int = 5,
                     min_size: tuple[int, int] = (40, 40)) -> List[Tuple[int, int, int, int]]:
        """Detect face bounding boxes in an image."""

        gray = self._gray(image)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        return [tuple(map(int, face)) for face in faces]

    @staticmethod
    def draw_faces(image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
                   color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """Annotate face detections on a copy of the image."""

        output = image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output,
                "Face",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output


class FaceRecognizer:
    """Optional LBPH-based face recognition wrapper.

    The recognizer requires opencv-contrib-python because cv2.face lives in the
    contrib package. The API is intentionally small: train from a directory of
    labeled subfolders, save/load a model, and predict labels for detected face
    crops.
    """

    def __init__(self, model_path: Optional[str] = None,
                 labels_path: Optional[str] = None,
                 detector: Optional[FaceDetector] = None) -> None:
        self.detector = detector or FaceDetector()
        self.model_path = model_path
        self.labels_path = labels_path
        self.label_to_name: Dict[int, str] = {}
        self.model = self._create_model()
        if model_path is not None:
            self.load(model_path, labels_path)

    @staticmethod
    def _create_model():
        if not hasattr(cv2, "face"):
            return None
        return cv2.face.LBPHFaceRecognizer_create()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _prepare_face_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        boxes = self.detector.detect_faces(image)
        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        gray = self.detector._gray(image)
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (200, 200))

    def train_from_directory(self, dataset_dir: str | Path,
                             save_model_path: Optional[str] = None,
                             save_labels_path: Optional[str] = None) -> None:
        """Train an LBPH recognizer from a labeled folder structure.

        The dataset directory should contain one subdirectory per identity.
        Each subdirectory may contain multiple face images. The directory names
        become the labels that are saved alongside the trained model.
        """

        if self.model is None:
            raise RuntimeError(
                "LBPH face recognition requires opencv-contrib-python with cv2.face"
            )

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        samples: List[np.ndarray] = []
        labels: List[int] = []
        label_to_name: Dict[int, str] = {}
        next_label = 0

        for identity_dir in sorted([p for p in dataset_path.iterdir() if p.is_dir()]):
            label_to_name[next_label] = identity_dir.name
            for image_path in sorted(identity_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                face_crop = self._prepare_face_crop(image)
                if face_crop is None:
                    gray = self.detector._gray(image)
                    face_crop = cv2.resize(gray, (200, 200))
                samples.append(face_crop)
                labels.append(next_label)
            next_label += 1

        if not samples:
            raise ValueError("No training faces were found in the dataset directory.")

        self.model.train(samples, np.asarray(labels, dtype=np.int32))
        self.label_to_name = label_to_name

        if save_model_path is not None:
            self.model.save(save_model_path)
        if save_labels_path is not None:
            with open(save_labels_path, "w", encoding="utf-8") as handle:
                json.dump(self.label_to_name, handle, indent=2)

    def load(self, model_path: str | Path,
             labels_path: Optional[str | Path] = None) -> None:
        """Load a trained LBPH model and optional label map from disk."""

        if self.model is None:
            raise RuntimeError(
                "LBPH face recognition requires opencv-contrib-python with cv2.face"
            )

        self.model.read(str(model_path))
        if labels_path is not None:
            with open(labels_path, "r", encoding="utf-8") as handle:
                raw_labels = json.load(handle)
            self.label_to_name = {int(key): value for key, value in raw_labels.items()}

    def predict(self, image: np.ndarray,
                confidence_threshold: float = 70.0) -> tuple[List[FacePrediction], np.ndarray]:
        """Detect faces and predict the identity for each crop."""

        if self.model is None:
            raise RuntimeError(
                "LBPH face recognition requires opencv-contrib-python with cv2.face"
            )

        output = image.copy()
        gray = self.detector._gray(image)
        predictions: List[FacePrediction] = []
        boxes = self.detector.detect_faces(image)

        for x, y, w, h in boxes:
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (200, 200))
            label_id, confidence = self.model.predict(crop)
            label = self.label_to_name.get(label_id, f"ID {label_id}")
            if confidence > confidence_threshold:
                label = "Unknown"
            predictions.append(
                FacePrediction(box=(x, y, w, h), label=label, confidence=float(confidence))
            )
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                output,
                f"{label} ({confidence:.1f})",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return predictions, output
