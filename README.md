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

## Installation

```bash
python -m pip install -r requirements.txt
```

The repository uses `opencv-contrib-python` so SIFT and the optional LBPH face recognizer are available.

If you plan to run the video pipeline on macOS and use a USB camera, make sure the terminal or editor has camera permissions.

## Usage

Run the still-image pipeline:

```bash
python main.py --image path/to/image.jpg --skip-video
```

Run image processing plus live camera analysis:

```bash
python main.py --image path/to/image.jpg --video-source 0
```

Run only the live video analyzer:

```bash
python main.py --skip-video --image path/to/image.jpg
```

Use `--video-source path/to/video.mp4` to analyze a file instead of a camera index.

If you have a trained LBPH model, pass `--face-model path/to/model.yml --labels path/to/labels.json` to enable recognition overlays.

## Notes

- SIFT may require an OpenCV build that includes the non-free/contrib modules.
- The LBPH face recognizer is optional and requires `cv2.face` from OpenCV contrib.
- HOG person detection uses the default pretrained pedestrian SVM provided by OpenCV.
- The image pipeline uses classical segmentation and feature methods rather than deep learning models, which keeps the repository lightweight and easy to extend.
