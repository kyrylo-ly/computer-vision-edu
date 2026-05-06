"""High-level pipeline orchestration for the computer vision demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2

from .classical_recognition import ClassicalFaceRecognizer
from .cnn_recognition import CNNFaceRecognizer
from .enhancement import ContrastEnhancer
from .face_recognition import FaceDetector, FaceRecognizer
from .feature_report import ImageAnalysisReport
from .features import FeatureExtractor, HumanDetector
from .geometry import GeometricTransformer
from .image_io import ImageLoader
from .morphology import MorphologyProcessor
from .preprocessing import ImagePreprocessor
from .segmentation import Segmenter
from .visualization import Visualizer


@dataclass
class PipelineOutputs:
    """Outputs produced by the still-image pipeline."""

    original: object
    denoised: object
    sharpened: object
    equalized: object
    clahe: object
    otsu_mask: object
    grabcut_mask: object
    grabcut_segmented: object
    watershed_mask: object
    watershed_segmented: object
    face_overlay: object
    face_boxes: object
    report: ImageAnalysisReport
    # Lab 4 – geometry outputs
    scaled_down: object
    scaled_up: object
    rotated: object
    perspective_demo: object
    # Lab 4 – morphology outputs
    morph_comparison: object
    morph_improved_mask: object


class CVPipeline:
    """Bundle the image and video stages into one workflow."""

    def __init__(self, image_path: Optional[str] = None,
                 face_model_path: Optional[str] = None,
                 labels_path: Optional[str] = None) -> None:
        self.image_path = image_path
        self.face_detector = FaceDetector()
        self.face_recognizer = None
        if face_model_path is not None:
            self.face_recognizer = FaceRecognizer(face_model_path, labels_path)

    def run_image_pipeline(self) -> PipelineOutputs:
        """Execute the still-image preprocessing and analysis workflow."""

        if self.image_path is None:
            raise ValueError("image_path must be provided for the image pipeline")

        image = ImageLoader.load_image(self.image_path)
        metadata = ImageLoader.get_metadata(image, self.image_path)
        ImageLoader.print_metadata(image, self.image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = ContrastEnhancer.compute_brightness_histogram(image)
        histogram_peak = int(histogram.bins[histogram.histogram.argmax()])
        print(f"Histogram peak intensity: {histogram_peak}")

        denoised = ImagePreprocessor.bilateral_denoise(image)
        sharpened = ImagePreprocessor.unsharp_mask(denoised)
        equalized = ContrastEnhancer.equalize_global_histogram(image)
        clahe = ContrastEnhancer.equalize_clahe(image)
        otsu_mask = Segmenter.otsu_binarize(gray)
        grabcut = Segmenter.grabcut_segment(image)
        watershed = Segmenter.watershed_segment(image)

        contours = HumanDetector.face_like_contours(otsu_mask)
        contour_vis = Visualizer.draw_contours(image, contours)
        feature_results = FeatureExtractor.compare_methods(image)
        for name, result in feature_results.items():
            print(f"{name.upper()}: {result['summary']}")

        face_boxes = self.face_detector.detect_faces(image)
        face_overlay = self.face_detector.draw_faces(image, face_boxes)
        if self.face_recognizer is not None and self.face_recognizer.available:
            _, face_overlay = self.face_recognizer.predict(image)

        # ----------------------------------------------------------------
        # Lab 4, Week 7 – Geometric transformations
        # ----------------------------------------------------------------
        print("\n--- Lab 4 / Week 7: Geometric Transformations ---")

        scale_down = GeometricTransformer.scale_by_factor(image, 0.5, 0.5, "area")
        scale_up   = GeometricTransformer.scale_by_factor(image, 1.5, 1.5, "cubic")
        print(
            f"Scale ×0.5 (AREA):  {scale_down.original_size} → {scale_down.new_size}"
        )
        print(
            f"Scale ×1.5 (CUBIC): {scale_up.original_size} → {scale_up.new_size}"
        )

        rotation_result = GeometricTransformer.rotate(image, angle_deg=15, expand=True)
        print(f"Rotation +15°: output size = {rotation_result.image.shape[:2][::-1]}"
              f", center = {rotation_result.center}")

        distorted, corrected = GeometricTransformer.demo_perspective(image)
        print("Perspective demo: distorted + corrected images generated.")

        # If any faces were detected, also warp the first face crop
        perspective_face = None
        if face_boxes:
            persp_result = GeometricTransformer.correct_face_perspective(
                image, face_boxes[0]
            )
            perspective_face = persp_result.image
            print(f"Face perspective crop size: {perspective_face.shape[:2][::-1]}")

        Visualizer.plot_image_grid(
            [
                image,
                scale_down.image,
                scale_up.image,
                rotation_result.image,
                distorted.image,
                corrected.image,
            ],
            [
                "Original",
                "Scale ×0.5 (AREA)",
                "Scale ×1.5 (CUBIC)",
                "Rotation +15° (expand)",
                "Perspective distorted",
                "Perspective corrected",
            ],
            cols=2,
            figsize=(16, 14),
        )

        # ----------------------------------------------------------------
        # Lab 4, Week 8 – Morphological operations
        # ----------------------------------------------------------------
        print("\n--- Lab 4 / Week 8: Morphological Operations ---")

        morph_comparison = MorphologyProcessor.compare_operations(
            otsu_mask, kernel_size=5, kernel_shape="ellipse", iterations=2
        )
        for r in morph_comparison.results:
            print(
                f"{r.operation.capitalize():<10} kernel={r.kernel_size}×{r.kernel_size}"
                f" ({r.kernel_shape}) ×{r.iterations}"
            )

        improved_mask, improve_comparison = MorphologyProcessor.improve_segmentation(
            otsu_mask, open_kernel=3, close_kernel=7, kernel_shape="ellipse"
        )
        print("Segmentation cleaned with open→close pipeline.")

        Visualizer.plot_image_grid(
            morph_comparison.as_image_list(),
            morph_comparison.as_title_list(),
            cols=3,
            figsize=(18, 10),
        )
        Visualizer.plot_image_grid(
            improve_comparison.as_image_list(),
            ["Raw Otsu mask", "After Opening (noise removal)", "After Closing (hole fill)"],
            cols=3,
            figsize=(18, 6),
        )

        # ----------------------------------------------------------------
        # Original visualisation grids (Labs 1-3)
        # ----------------------------------------------------------------
        Visualizer.plot_image_grid(
            [image, denoised, sharpened, equalized, clahe],
            ["Original", "Denoised", "Sharpened", "Global Equalization", "CLAHE"],
            cols=2,
        )
        Visualizer.plot_image_grid(
            [
                otsu_mask,
                grabcut.mask,
                grabcut.segmented_image,
                watershed.mask,
                watershed.segmented_image,
                contour_vis,
                face_overlay,
            ],
            [
                "Otsu Mask",
                "GrabCut Mask",
                "GrabCut Segmented",
                "Watershed Mask",
                "Watershed Segmented",
                "Contour Overlay",
                "Face Overlay",
            ],
            cols=2,
            figsize=(16, 14),
        )

        report = ImageAnalysisReport(
            metadata=metadata.__dict__,
            histogram_peak=histogram_peak,
            face_boxes=face_boxes,
            feature_summaries={
                name: result["summary"] for name, result in feature_results.items()
            },
        )

        return PipelineOutputs(
            original=image,
            denoised=denoised,
            sharpened=sharpened,
            equalized=equalized,
            clahe=clahe,
            otsu_mask=otsu_mask,
            grabcut_mask=grabcut.mask,
            grabcut_segmented=grabcut.segmented_image,
            watershed_mask=watershed.mask,
            watershed_segmented=watershed.segmented_image,
            face_overlay=face_overlay,
            face_boxes=face_boxes,
            report=report,
            scaled_down=scale_down.image,
            scaled_up=scale_up.image,
            rotated=rotation_result.image,
            perspective_demo=corrected.image,
            morph_comparison=morph_comparison,
            morph_improved_mask=improved_mask,
        )

    # ------------------------------------------------------------------
    # Lab 5 pipeline
    # ------------------------------------------------------------------

    def run_lab5_pipeline(
        self,
        dataset_dir: Optional[str] = None,
        cnn_epochs: int = 20,
        test_size: float = 0.25,
    ) -> dict:
        """Execute Lab 5: classical ML + CNN face recognition.

        When *dataset_dir* is provided the recognizers are trained on real
        face images (one sub-folder per identity).  When it is ``None`` the
        method generates a small synthetic dataset so the full pipeline can
        be demonstrated and verified without external data.

        Parameters
        ----------
        dataset_dir:
            Path to a directory whose sub-folders each contain face images
            of one identity.  Pass ``None`` to use synthetic data.
        cnn_epochs:
            Maximum number of training epochs for the CNN.
        test_size:
            Fraction of images reserved for evaluation.

        Returns
        -------
        dict
            Keys ``"classical_report"`` and ``"cnn_history"`` with the
            corresponding result objects.
        """
        import tempfile, shutil, pathlib
        import numpy as np

        synthetic = dataset_dir is None
        tmp_dir = None

        if synthetic:
            print("No dataset_dir provided — generating synthetic face data …")
            tmp_dir = tempfile.mkdtemp(prefix="cv_lab5_")
            dataset_dir = tmp_dir
            _make_synthetic_dataset(tmp_dir, n_classes=4, images_per_class=20)
            print(f"  Synthetic dataset written to {tmp_dir}")

        results: dict = {}

        # ---- Week 9: Classical classifiers --------------------------------
        print("\n=== Lab 5 / Week 9: Classical ML Face Recognition ===")
        classical = ClassicalFaceRecognizer(target_size=(64, 64))
        try:
            report = classical.train_from_directory(
                dataset_dir, test_size=test_size
            )
            print(report.summary())
            results["classical_report"] = report
        except Exception as exc:
            print(f"Classical pipeline error: {exc}")
            results["classical_report"] = None

        # ---- Week 10: CNN -------------------------------------------------
        print("\n=== Lab 5 / Week 10: CNN Face Recognition ===")
        cnn = CNNFaceRecognizer(input_size=(64, 64))
        try:
            history = cnn.train_from_directory(
                dataset_dir,
                epochs=cnn_epochs,
                test_size=test_size,
            )
            cnn.plot_history()
            results["cnn_history"] = history
        except Exception as exc:
            print(f"CNN pipeline error: {exc}")
            results["cnn_history"] = None

        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return results


# ---------------------------------------------------------------------------
# Synthetic dataset helper
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    root: str,
    n_classes: int = 4,
    images_per_class: int = 20,
    image_size: int = 96,
) -> None:
    """Write random grey face-like patches to a temporary directory.

    Each class gets a deterministic base colour so the classifiers have a
    learnable signal even though the images are noise.
    """
    import os
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    for cls_id in range(n_classes):
        cls_dir = os.path.join(root, f"person_{cls_id:02d}")
        os.makedirs(cls_dir, exist_ok=True)
        base = int(40 + cls_id * 50)
        for img_idx in range(images_per_class):
            patch = rng.integers(
                max(0, base - 30), min(255, base + 30),
                size=(image_size, image_size), dtype=np.uint8,
            )
            # Add a rough ellipse to mimic a face silhouette
            cx, cy = image_size // 2, image_size // 2
            cv2.ellipse(
                patch, (cx, cy),
                (cx - 10, cy - 8), 0, 0, 360,
                color=int(np.clip(base + 20, 0, 255)), thickness=-1,
            )
            out_path = os.path.join(cls_dir, f"img_{img_idx:03d}.png")
            cv2.imwrite(out_path, patch)
