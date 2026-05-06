"""Computer vision utilities for face and human search workflows."""

from .classical_recognition import ClassicalFaceRecognizer, ComparisonReport
from .cnn_recognition import CNNFaceRecognizer, TrainingHistory
from .enhancement import ContrastEnhancer, compute_brightness_histogram
from .face_recognition import FaceDetector, FaceRecognizer
from .features import FeatureExtractor, HumanDetector
from .geometry import GeometricTransformer
from .image_io import ImageLoader, ImageMetadata
from .morphology import KernelFactory, MorphologyProcessor
from .preprocessing import ImagePreprocessor
from .segmentation import Segmenter
from .video import VideoAnalyzer

__all__ = [
    "ClassicalFaceRecognizer",
    "CNNFaceRecognizer",
    "ComparisonReport",
    "ContrastEnhancer",
    "compute_brightness_histogram",
    "FaceDetector",
    "FaceRecognizer",
    "FeatureExtractor",
    "GeometricTransformer",
    "HumanDetector",
    "ImageLoader",
    "ImageMetadata",
    "ImagePreprocessor",
    "KernelFactory",
    "MorphologyProcessor",
    "Segmenter",
    "TrainingHistory",
    "VideoAnalyzer",
]
