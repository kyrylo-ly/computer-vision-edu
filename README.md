# Computer Vision Face and Human Search Demo

A modular classical-computer-vision pipeline for still-image preprocessing, optional face recognition, and real-time video analysis.

## Features

- Robust image loading for JPG, PNG, and BMP via Pillow and OpenCV
- Image diagnostics with resolution, channels, and data type reporting
- Brightness histograms and contrast enhancement with global equalization and CLAHE
- Noise reduction with Gaussian, Median, and Bilateral filtering
- Sharpening with Laplacian and Unsharp Masking
- Segmentation with Otsu thresholding, GrabCut, and Watershed
- Face detection with Haar cascades and optional LBPH recognition
- Contour extraction plus SIFT, ORB, and HOG-based feature analysis
- Real-time background subtraction with MOG2 or KNN
- Optical flow motion estimation with Lucas-Kanade or Farneback
- **[Lab 4]** Geometric transformations: scaling (multiple interpolations), rotation with canvas expansion, perspective warp and face-crop correction
- **[Lab 4]** Morphological operations: erosion, dilation, opening, closing — with kernel-size sweep and open→close segmentation improvement pipeline

## Module overview

| Module | Lab | Description |
|---|---|---|
| `image_io.py` | 1 | `ImageLoader` — Pillow-based loading, metadata, BGR↔RGB helpers |
| `enhancement.py` | 1 | `ContrastEnhancer` — brightness histogram, global EQ, CLAHE |
| `preprocessing.py` | 2 | `ImagePreprocessor` — Gaussian / Median / Bilateral, Laplacian / Unsharp |
| `segmentation.py` | 2 | `Segmenter` — Otsu, GrabCut, Watershed |
| `features.py` | 3 | `FeatureExtractor` — SIFT, ORB, HOG; `HumanDetector` — contours |
| `face_recognition.py` | 3 | `FaceDetector` (Haar), `FaceRecognizer` (LBPH, optional) |
| `video.py` | 3 | `VideoAnalyzer` — MOG2/KNN background subtraction, LK/Farneback optical flow |
| `geometry.py` | **4** | `GeometricTransformer` — scaling, rotation, perspective warp |
| `morphology.py` | **4** | `MorphologyProcessor` — erosion, dilation, opening, closing |
| `visualization.py` | all | `Visualizer` — Matplotlib grids, histogram plots, contour/box overlays |
| `pipeline.py` | all | `CVPipeline` — orchestrates all stages end-to-end |

## Installation

```bash
python -m pip install -r requirements.txt
```

The repository uses `opencv-contrib-python` so SIFT and the optional LBPH face recognizer are available.

If you plan to run the video pipeline on macOS and use a USB camera, make sure the terminal or editor has camera permissions.

## Usage

Run the still-image pipeline (all labs):

```bash
python main.py --image path/to/image.jpg --skip-video
```

Run image processing plus live camera analysis:

```bash
python main.py --image path/to/image.jpg --video-source 0
```

Run only the live video analyzer:

```bash
python main.py --video-source 0
```

Use `--video-source path/to/video.mp4` to analyze a file instead of a camera index.

If you have a trained LBPH model, pass `--face-model path/to/model.yml --labels path/to/labels.json` to enable recognition overlays.

## Lab 4 highlights

### Geometric transformations (`geometry.py`)

```python
from cv_human_search.geometry import GeometricTransformer

# Scaling with explicit interpolation
r = GeometricTransformer.scale_by_factor(image, 0.5, 0.5, interpolation="area")
r = GeometricTransformer.scale_to_size(image, 128, 128, interpolation="cubic")

# Rotation (expand=True keeps full content without clipping)
r = GeometricTransformer.rotate(image, angle_deg=15, expand=True)

# Perspective warp demo (distortion + correction pair)
distorted, corrected = GeometricTransformer.demo_perspective(image)

# Normalise a detected face crop
r = GeometricTransformer.correct_face_perspective(image, face_box=(x, y, w, h))
```

### Morphological operations (`morphology.py`)

```python
from cv_human_search.morphology import MorphologyProcessor

# Primitive operations on a binary mask
e = MorphologyProcessor.erode(mask, kernel_size=5, kernel_shape="ellipse")
d = MorphologyProcessor.dilate(mask, kernel_size=5)
o = MorphologyProcessor.open(mask, kernel_size=3, iterations=2)
c = MorphologyProcessor.close(mask, kernel_size=7, iterations=2)

# All four operations in one comparison object (ready for Visualizer)
cmp = MorphologyProcessor.compare_operations(mask, kernel_size=5)

# Two-stage open→close segmentation cleaner
cleaned_mask, steps = MorphologyProcessor.improve_segmentation(mask)
```

## Notes

- SIFT may require an OpenCV build that includes the non-free/contrib modules.
- The LBPH face recognizer is optional and requires `cv2.face` from OpenCV contrib.
- HOG person detection uses the default pretrained pedestrian SVM provided by OpenCV.
- SURF is not included because it is patent-restricted and unavailable in standard OpenCV builds; SIFT and ORB cover the same role.
- The image pipeline uses classical segmentation and feature methods rather than deep learning models, which keeps the repository lightweight and easy to extend.
