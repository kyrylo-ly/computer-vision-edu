"""Classical ML face recognition: SVM, Random Forest, KNN (Lab 5, Week 9).

Pipeline: detect face → crop → flatten → StandardScaler → PCA → classifier.
Three classifiers are trained on the same features and compared side-by-side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ClassifierMetrics:
    """Evaluation metrics for one classifier."""
    name: str
    accuracy: float
    report_text: str
    confusion: np.ndarray


@dataclass
class ComparisonReport:
    """Aggregated comparison of all three classifiers."""
    metrics: List[ClassifierMetrics] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_components: int = 0

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.n_train} train / {self.n_test} test",
            f"PCA components: {self.n_components}",
            f"Classes: {self.label_names}",
            "",
        ]
        for m in self.metrics:
            lines.append("─" * 50)
            lines.append(f"  {m.name}  —  accuracy = {m.accuracy:.4f}")
            lines.append(m.report_text)
        return "\n".join(lines)

    def plot_comparison(self) -> None:
        """Plot accuracy bar-chart and confusion matrices for all classifiers."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            print("matplotlib is required for plot_comparison().")
            return

        n = len(self.metrics)
        fig = plt.figure(figsize=(6 + n * 4, 10))
        gs  = fig.add_gridspec(2, n + 1, hspace=0.45, wspace=0.35)

        # ── top row: accuracy bar chart (spans all columns) ──────────────
        ax_bar = fig.add_subplot(gs[0, :])
        names  = [m.name for m in self.metrics]
        accs   = [m.accuracy for m in self.metrics]
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        bars   = ax_bar.bar(names, accs, color=colors[: len(names)], width=0.45,
                            edgecolor="white", linewidth=0.8)
        for bar, acc in zip(bars, accs):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                acc + 0.012,
                f"{acc:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax_bar.set_ylim(0, 1.12)
        ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax_bar.set_title(
            f"Classifier Accuracy Comparison\n"
            f"(train={self.n_train}, test={self.n_test}, PCA={self.n_components})",
            fontsize=13,
        )
        ax_bar.set_ylabel("Accuracy")
        ax_bar.grid(axis="y", alpha=0.35)
        ax_bar.spines[["top", "right"]].set_visible(False)

        # ── bottom row: one confusion matrix per classifier ───────────────
        for col, m in enumerate(self.metrics):
            ax = fig.add_subplot(gs[1, col])
            cm = m.confusion.astype(float)
            # Normalise by row so colours are comparable across class sizes
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums

            im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(self.label_names)))
            ax.set_yticks(range(len(self.label_names)))
            ax.set_xticklabels(self.label_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(self.label_names, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{m.name}\nacc = {m.accuracy:.3f}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate cells with raw counts
            for i in range(len(self.label_names)):
                for j in range(len(self.label_names)):
                    val = int(m.confusion[i, j])
                    color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=8, color=color)

        plt.suptitle("Lab 5 — Classical ML Face Recognition", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()



# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FaceFeatureExtractor:
    """Crop face, resize, flatten to a fixed-length float32 vector."""

    def __init__(self, target_size: Tuple[int, int] = (64, 64)) -> None:
        self.target_size = target_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def _gray(self, img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _largest_face(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        return None if len(faces) == 0 else max(faces, key=lambda f: f[2] * f[3])

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Return a flat float32 vector normalised to [0, 1]."""
        gray = self._gray(image)
        box = self._largest_face(gray)
        crop = gray[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] if box else gray
        resized = cv2.resize(crop, self.target_size)
        return resized.flatten().astype(np.float32) / 255.0

    @property
    def feature_size(self) -> int:
        return self.target_size[0] * self.target_size[1]


# ---------------------------------------------------------------------------
# Main recognizer
# ---------------------------------------------------------------------------

class ClassicalFaceRecognizer:
    """Train SVM, Random Forest, and KNN on face images and compare results.

    Dataset layout expected::

        dataset_dir/
            Alice/  image1.jpg  image2.jpg  ...
            Bob/    image1.jpg  ...

    Usage::

        rec = ClassicalFaceRecognizer()
        report = rec.train_from_directory("data/faces/")
        print(report.summary())
        label, prob = rec.predict(image, classifier_name="SVM (RBF)")
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        n_neighbors: int = 5,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )
        self.extractor = FaceFeatureExtractor(target_size=target_size)
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.is_trained = False

        self.classifiers: Dict[str, object] = {
            "SVM (RBF)": SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=n_neighbors, metric="euclidean", weights="distance",
            ),
        }

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(
        self, dataset_dir: str | Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load images; each sub-directory = one identity class."""
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not subdirs:
            raise ValueError(f"No identity sub-directories found in {root}.")

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        X_list, y_list = [], []
        for label_id, d in enumerate(subdirs):
            self.label_to_name[label_id] = d.name
            self.name_to_label[d.name] = label_id
            for p in d.iterdir():
                if p.suffix.lower() not in exts:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                X_list.append(self.extractor.extract(img))
                y_list.append(label_id)

        if not X_list:
            raise ValueError("No valid images found in the dataset.")
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X_train)
        max_c = min(X_train.shape[0] - 1, X_train.shape[1])
        self.pca = PCA(n_components=min(150, max_c), random_state=42)
        return self.pca.fit_transform(X_scaled)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_directory(
        self,
        dataset_dir: str | Path,
        test_size: float = 0.25,
    ) -> ComparisonReport:
        """Load → extract features → train all classifiers → evaluate.

        Parameters
        ----------
        dataset_dir:
            Root with one sub-folder per identity.
        test_size:
            Fraction held out for evaluation (default 0.25).
        """
        print(f"Loading dataset from '{dataset_dir}' …")
        X, y = self._load_dataset(dataset_dir)
        print(f"  {len(X)} samples, {len(self.label_to_name)} classes.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        print("Fitting scaler + PCA …")
        X_train_pca = self._fit_transform(X_train)
        X_test_pca = self._transform(X_test)
        n_comp = X_train_pca.shape[1]
        print(f"  {X.shape[1]} → {n_comp} PCA components.")

        label_names = [self.label_to_name[i] for i in sorted(self.label_to_name)]
        report = ComparisonReport(
            label_names=label_names,
            n_train=len(X_train),
            n_test=len(X_test),
            n_components=n_comp,
        )

        for clf_name, clf in self.classifiers.items():
            print(f"Training {clf_name} …")
            clf.fit(X_train_pca, y_train)
            y_pred = clf.predict(X_test_pca)
            acc = float(np.mean(y_pred == y_test))
            rep = classification_report(
                y_test, y_pred, target_names=label_names, zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred)
            report.metrics.append(
                ClassifierMetrics(name=clf_name, accuracy=acc,
                                  report_text=rep, confusion=cm)
            )
            print(f"  accuracy = {acc:.4f}")

        self.is_trained = True
        return report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        classifier_name: str = "SVM (RBF)",
    ) -> Tuple[str, float]:
        """Predict identity for a single BGR image.

        Returns ``(label_string, probability)``.
        """
        if not self.is_trained:
            raise RuntimeError("Call train_from_directory() first.")
        clf = self.classifiers[classifier_name]
        feat = self.extractor.extract(image).reshape(1, -1)
        feat_pca = self._transform(feat)
        label_id = int(clf.predict(feat_pca)[0])
        prob = float(clf.predict_proba(feat_pca).max())
        return self.label_to_name.get(label_id, f"ID {label_id}"), prob

    def predict_all(self, image: np.ndarray) -> Dict[str, Tuple[str, float]]:
        """Run all three classifiers on the same image."""
        return {name: self.predict(image, name) for name in self.classifiers}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        """Pickle preprocessors + classifiers to *output_dir*."""
        import pickle
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(out / "pca.pkl", "wb") as f:
            pickle.dump(self.pca, f)
        with open(out / "classifiers.pkl", "wb") as f:
            pickle.dump(self.classifiers, f)
        with open(out / "labels.json", "w") as f:
            json.dump(self.label_to_name, f, indent=2)
        print(f"Saved classical recognizer to {out}/")

    def load(self, model_dir: str | Path) -> None:
        """Restore a previously saved recognizer."""
        import pickle
        d = Path(model_dir)
        with open(d / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(d / "pca.pkl", "rb") as f:
            self.pca = pickle.load(f)
        with open(d / "classifiers.pkl", "rb") as f:
            self.classifiers = pickle.load(f)
        with open(d / "labels.json") as f:
            raw = json.load(f)
        self.label_to_name = {int(k): v for k, v in raw.items()}
        self.name_to_label = {v: k for k, v in self.label_to_name.items()}
        self.is_trained = True
