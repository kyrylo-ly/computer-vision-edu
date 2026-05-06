"""Command-line entrypoint for the computer vision demo."""
# Supports Lab 1-4 still-image + video pipeline and Lab 5 face recognition.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cv_human_search.pipeline import CVPipeline
from cv_human_search.video import VideoAnalyzer


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Face recognition and human searching demo using classical "
            "computer vision methods."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a JPG, PNG, or BMP image used for the still pipeline.",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default="0",
        help="Camera index or video file path for live analysis.",
    )
    parser.add_argument(
        "--bg-method",
        choices=("mog2", "knn"),
        default="mog2",
        help="Background subtraction method.",
    )
    parser.add_argument(
        "--flow-method",
        choices=("lk", "farneback"),
        default="lk",
        help="Optical flow method.",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default=None,
        help="Optional LBPH face model path for recognition.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional label mapping JSON for the LBPH face model.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Run only the image pipeline.",
    )
    parser.add_argument(
        "--lab5",
        action="store_true",
        help=(
            "Run the Lab 5 face-recognition pipeline "
            "(classical ML + CNN). Skips the still-image and video pipelines."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to a face dataset directory for --lab5. "
            "Each sub-folder should contain images of one identity. "
            "If omitted, a small synthetic dataset is generated automatically."
        ),
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=20,
        metavar="N",
        help="Maximum training epochs for the CNN in --lab5 (default: 20).",
    )
    return parser


def parse_video_source(value: str):
    """Interpret the video source as a camera index or file path."""

    if value.isdigit():
        return int(value)
    return value


def main() -> int:
    """Run the still-image pipeline and optionally the live video demo."""

    parser = build_parser()
    args = parser.parse_args()

    # ---- Lab 5: face recognition (classical ML + CNN) -------------------
    if args.lab5:
        # CVPipeline.__init__ accepts image_path=None, so no image is needed.
        CVPipeline().run_lab5_pipeline(
            dataset_dir=args.dataset,
            cnn_epochs=args.cnn_epochs,
        )
        return 0

    if args.image is None and args.skip_video:
        parser.error("Provide --image when using --skip-video.")

    if args.image is not None:
        pipeline = CVPipeline(
            args.image,
            face_model_path=args.face_model,
            labels_path=args.labels,
        )
        pipeline.run_image_pipeline()

    if not args.skip_video:
        analyzer = VideoAnalyzer(
            source=parse_video_source(args.video_source),
            bg_method=args.bg_method,
            flow_method=args.flow_method,
            display=True,
        )
        analyzer.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
