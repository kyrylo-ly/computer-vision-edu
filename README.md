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
- **[Lab 5, Week 9]** Classical ML face recognition: SVM (RBF), Random Forest, and KNN trained on PCA-compressed face embeddings with per-classifier accuracy comparison
- **[Lab 5, Week 10]** CNN face recognition: custom TensorFlow/Keras convolutional network with data augmentation, early stopping, training-curve plots, and feature-map visualisation

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
| `classical_recognition.py` | **5** | `ClassicalFaceRecognizer` — SVM / Random Forest / KNN on PCA features |
| `cnn_recognition.py` | **5** | `CNNFaceRecognizer` — TensorFlow/Keras CNN with augmentation and history plots |
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

Run the Lab 5 face-recognition pipeline with a synthetic dataset (no data required):

```bash
python main.py --lab5
```

Run Lab 5 with a real face dataset:

```bash
python main.py --lab5 --dataset path/to/faces/ --cnn-epochs 30
```

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

## Lab 5 highlights

### Week 9 — Classical ML face recognition (`classical_recognition.py`)

Three classifiers are trained on the same feature set and compared side-by-side.
Feature extraction pipeline: detect face → crop → resize to 64 × 64 → flatten → StandardScaler → PCA (up to 150 components).

```python
from cv_human_search.classical_recognition import ClassicalFaceRecognizer

rec = ClassicalFaceRecognizer(target_size=(64, 64), n_neighbors=5, n_estimators=150)
report = rec.train_from_directory("data/faces/")   # one sub-folder per identity
print(report.summary())                            # accuracy table for all three models

# Single-image inference
label, prob = rec.predict(image, classifier_name="SVM (RBF)")
label, prob = rec.predict(image, classifier_name="Random Forest")
label, prob = rec.predict(image, classifier_name="KNN")

# Run all three classifiers at once
results = rec.predict_all(image)   # {"SVM (RBF)": (label, prob), ...}

# Persist / restore
rec.save("models/classical/")
rec.load("models/classical/")
```

### Week 10 — CNN face recognition (`cnn_recognition.py`)

Custom CNN trained end-to-end with `tf.data` pipelines and image augmentation.
Architecture: Conv→BN→ReLU blocks × 3 → GlobalAvgPool → Dropout → Dense(softmax).

```python
from cv_human_search.cnn_recognition import CNNFaceRecognizer

cnn = CNNFaceRecognizer(input_size=(64, 64))
history = cnn.train_from_directory(
    "data/faces/",
    epochs=30,
    batch_size=32,
    patience=8,
)
print(history.summary())

# Plot training/validation loss & accuracy curves
cnn.plot_history()

# Visualise feature maps of a chosen convolutional layer
cnn.visualize_feature_maps(image, layer_name="conv1")

# Inference
label, prob   = cnn.predict(image)          # top-1
topk          = cnn.predict_topk(image, k=3)  # top-3 list of (label, prob)

# Persist / restore
cnn.save("models/cnn/")
cnn.load("models/cnn/")
```

### Running the full Lab 5 pipeline from Python

```python
from cv_human_search.pipeline import CVPipeline

pipeline = CVPipeline()

# Without a real dataset — synthetic data is generated automatically
results = pipeline.run_lab5_pipeline()

# With a real face dataset
results = pipeline.run_lab5_pipeline(
    dataset_dir="data/faces/",
    cnn_epochs=30,
    test_size=0.2,
)

classical_report = results["classical_report"]  # ComparisonReport
cnn_history      = results["cnn_history"]        # TrainingHistory
```

## Notes

- SIFT may require an OpenCV build that includes the non-free/contrib modules.
- The LBPH face recognizer is optional and requires `cv2.face` from OpenCV contrib.
- HOG person detection uses the default pretrained pedestrian SVM provided by OpenCV.
- SURF is not included because it is patent-restricted and unavailable in standard OpenCV builds; SIFT and ORB cover the same role.
- The image pipeline uses classical segmentation and feature methods rather than deep learning models, which keeps the repository lightweight and easy to extend.
