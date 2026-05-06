"""High-level pipeline orchestration for the computer vision demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2

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

