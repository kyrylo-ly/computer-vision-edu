"""CNN face recognition using TensorFlow/Keras (Lab 5, Week 10).

Architecture (LeNet-inspired, adapted for small face datasets)
--------------------------------------------------------------
Input 64×64×1 greyscale
  → Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool
  → Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool
  → Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool
  → GlobalAveragePooling
  → Dense(256, ReLU) → Dropout(0.5)
  → Dense(N, Softmax)

Data augmentation (applied only during training)
-------------------------------------------------
Random horizontal flip, rotation ±15°, zoom ±10 %, brightness ±20 %.

Usage
-----
::

    cnn = CNNFaceRecognizer(input_size=(64, 64))
    history = cnn.train_from_directory("data/faces/", epochs=30)
    cnn.plot_history()
    label, prob = cnn.predict(image)
    cnn.save("models/cnn_face/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Stores epoch-by-epoch loss and accuracy for train and validation."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def best_val_accuracy(self) -> float:
        return max(self.val_acc) if self.val_acc else 0.0

    def summary(self) -> str:
        epochs = len(self.train_acc)
        if epochs == 0:
            return "No training history."
        best_acc = self.best_val_accuracy()
        best_ep = self.val_acc.index(best_acc) + 1
        final_loss = self.val_loss[-1]
        return (
            f"Epochs: {epochs}  |  "
            f"Best val_acc: {best_acc:.4f} (epoch {best_ep})  |  "
            f"Final val_loss: {final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Dataset loader (reused by classical module pattern)
# ---------------------------------------------------------------------------

class _DatasetLoader:
    """Load face images into numpy arrays suitable for Keras."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, input_size: Tuple[int, int]) -> None:
        self.input_size = input_size
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Detect face, crop, resize to input_size, normalise to [0, 1]."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            gray = gray[y: y + h, x: x + w]
        resized = cv2.resize(gray, self.input_size)
        return resized.astype(np.float32) / 255.0

    def load(
        self, dataset_dir: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """Return ``(X, y, label_to_name)``."""
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not subdirs:
            raise ValueError(f"No sub-directories in {root}.")

        label_to_name: Dict[int, str] = {}
        X_list, y_list = [], []
        for label_id, d in enumerate(subdirs):
            label_to_name[label_id] = d.name
            for p in d.iterdir():
                if p.suffix.lower() not in self.EXTENSIONS:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                X_list.append(self._preprocess(img))
                y_list.append(label_id)

        if not X_list:
            raise ValueError("No valid images found.")
        X = np.stack(X_list)[..., np.newaxis]   # shape: (N, H, W, 1)
        y = np.array(y_list, dtype=np.int32)
        return X, y, label_to_name


# ---------------------------------------------------------------------------
# CNN recognizer
# ---------------------------------------------------------------------------

class CNNFaceRecognizer:
    """Convolutional Neural Network for face identity classification.

    The model is built with TensorFlow/Keras.  Data augmentation is applied
    during training to improve generalisation on small datasets.

    Parameters
    ----------
    input_size:
        ``(height, width)`` of the face patches fed into the network.
    """

    def __init__(self, input_size: Tuple[int, int] = (64, 64)) -> None:
        if not _TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )
        self.input_size = input_size
        self.label_to_name: Dict[int, str] = {}
        self.num_classes: int = 0
        self.model: Optional[keras.Model] = None
        self.history: Optional[TrainingHistory] = None
        self._loader = _DatasetLoader(input_size)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, num_classes: int) -> keras.Model:
        """Construct the CNN architecture.

        The network uses three convolutional blocks followed by global
        average pooling and a fully-connected head.  Batch normalisation
        after each convolution speeds up convergence; dropout before the
        output layer reduces over-fitting on small datasets.
        """
        inp = keras.Input(shape=(*self.input_size, 1), name="face_input")

        # Block 1 — low-level edges
        x = layers.Conv2D(32, 3, padding="same", activation="relu",
                          name="conv1")(inp)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.MaxPooling2D(name="pool1")(x)

        # Block 2 — mid-level textures
        x = layers.Conv2D(64, 3, padding="same", activation="relu",
                          name="conv2")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.MaxPooling2D(name="pool2")(x)

        # Block 3 — high-level facial features
        x = layers.Conv2D(128, 3, padding="same", activation="relu",
                          name="conv3")(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.MaxPooling2D(name="pool3")(x)

        # Head
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout")(x)
        out = layers.Dense(num_classes, activation="softmax",
                           name="output")(x)

        model = keras.Model(inputs=inp, outputs=out, name="CNNFaceRecognizer")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _build_augmentation(self) -> keras.Sequential:
        """Return a Keras Sequential model that applies random augmentations."""
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),   # ± ~15°
                layers.RandomZoom(0.10),
                layers.RandomBrightness(0.20),
            ],
            name="augmentation",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_directory(
        self,
        dataset_dir: str | Path,
        test_size: float = 0.20,
        epochs: int = 30,
        batch_size: int = 16,
        patience: int = 8,
    ) -> TrainingHistory:
        """Load dataset, build model, train, and return history.

        Parameters
        ----------
        dataset_dir:
            Root with one sub-folder per identity.
        test_size:
            Fraction of data used for validation.
        epochs:
            Maximum number of training epochs.
        batch_size:
            Mini-batch size.
        patience:
            Early-stopping patience (epochs without val_loss improvement).

        Returns
        -------
        TrainingHistory
            Per-epoch loss and accuracy for train and validation splits.
        """
        from sklearn.model_selection import train_test_split

        print(f"Loading dataset from '{dataset_dir}' …")
        X, y, label_to_name = self._loader.load(dataset_dir)
        self.label_to_name = label_to_name
        self.num_classes = len(label_to_name)
        print(f"  {len(X)} samples, {self.num_classes} classes.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Build augmentation + model
        augment = self._build_augmentation()
        self.model = self._build_model(self.num_classes)
        self.model.summary(print_fn=print)

        # Build tf.data pipelines
        def make_dataset(Xd, yd, training: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((Xd, yd))
            if training:
                ds = ds.shuffle(len(Xd), seed=42)
                ds = ds.map(
                    lambda x, y_: (augment(x[tf.newaxis], training=True)[0], y_),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = make_dataset(X_train, y_train, training=True)
        val_ds   = make_dataset(X_val,   y_val,   training=False)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, verbose=1,
            ),
        ]

        print(f"\nTraining CNN for up to {epochs} epochs …")
        keras_hist = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        h = keras_hist.history
        self.history = TrainingHistory(
            train_loss=h["loss"],
            val_loss=h["val_loss"],
            train_acc=h["accuracy"],
            val_acc=h["val_accuracy"],
        )
        print(f"\n{self.history.summary()}")
        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Crop face and prepare the image for inference."""
        return self._loader._preprocess(image)[np.newaxis, ..., np.newaxis]

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Return ``(label_string, probability)`` for the top-1 prediction."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train_from_directory() first.")
        x = self.preprocess_image(image)
        probs = self.model.predict(x, verbose=0)[0]
        label_id = int(np.argmax(probs))
        return self.label_to_name.get(label_id, f"ID {label_id}"), float(probs[label_id])

    def predict_topk(
        self, image: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Return top-k ``(label, probability)`` predictions."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        x = self.preprocess_image(image)
        probs = self.model.predict(x, verbose=0)[0]
        top_ids = np.argsort(probs)[::-1][:k]
        return [
            (self.label_to_name.get(int(i), f"ID {i}"), float(probs[i]))
            for i in top_ids
        ]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_history(self) -> None:
        """Plot training and validation loss + accuracy curves."""
        if self.history is None:
            raise RuntimeError("No history. Train the model first.")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plot_history().")
            return

        epochs = range(1, len(self.history.train_loss) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(epochs, self.history.train_loss, label="Train loss")
        ax1.plot(epochs, self.history.val_loss,   label="Val loss")
        ax1.set_title("Loss per epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Sparse categorical cross-entropy")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, self.history.train_acc, label="Train accuracy")
        ax2.plot(epochs, self.history.val_acc,   label="Val accuracy")
        ax2.set_title("Accuracy per epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle("CNN Face Recognizer — Training History")
        plt.tight_layout()
        plt.show()

    def visualize_feature_maps(
        self, image: np.ndarray, layer_name: str = "conv1"
    ) -> None:
        """Display the activation maps of a chosen convolutional layer.

        This reveals which image regions the network responds to, helping
        understand what the CNN has learned.

        Parameters
        ----------
        image:
            Raw BGR image containing a face.
        layer_name:
            Name of the Conv2D layer to inspect (e.g. ``"conv1"``,
            ``"conv2"``, ``"conv3"``).
        """
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualize_feature_maps().")
            return

        # Build a sub-model that outputs the chosen layer
        try:
            layer_output = self.model.get_layer(layer_name).output
        except ValueError:
            available = [l.name for l in self.model.layers]
            raise ValueError(
                f"Layer '{layer_name}' not found. Available: {available}"
            )
        vis_model = keras.Model(inputs=self.model.input, outputs=layer_output)

        x = self.preprocess_image(image)
        feature_maps = vis_model.predict(x, verbose=0)[0]  # (H, W, C)

        n_maps = min(16, feature_maps.shape[-1])
        cols = 4
        rows = (n_maps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).ravel()
        for i in range(n_maps):
            axes[i].imshow(feature_maps[..., i], cmap="viridis")
            axes[i].set_title(f"Filter {i}")
            axes[i].axis("off")
        for i in range(n_maps, len(axes)):
            axes[i].axis("off")
        plt.suptitle(f"Feature maps — layer '{layer_name}'")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        """Save the Keras model and label map to *output_dir*."""
        if self.model is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out / "model.keras"))
        with open(out / "labels.json", "w") as f:
            json.dump(self.label_to_name, f, indent=2)
        print(f"CNN model saved to {out}/")

    def load(self, model_dir: str | Path) -> None:
        """Restore a previously saved CNN recognizer."""
        d = Path(model_dir)
        self.model = keras.models.load_model(str(d / "model.keras"))
        with open(d / "labels.json") as f:
            raw = json.load(f)
        self.label_to_name = {int(k): v for k, v in raw.items()}
        self.num_classes = len(self.label_to_name)
        print(f"CNN model loaded from {d}/")
