"""HOG + Linear SVM model for facial expression recognition (FER-2013).

Usage:
    import svm_model
    emotion, confidence, ms = svm_model.predict(face)

The model (StandardScaler + LinearSVC, one-vs-rest) is trained in TP_HOG.ipynb.
"""
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

MODEL_PATH = Path(__file__).parent / "hog_linear_svm.joblib"
_model = joblib.load(MODEL_PATH)


def extract_features(face):
    """48x48 grayscale face -> HOG vector (900 values)."""
    return hog(
        face,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        block_norm="L2-Hys",
    )


def predict(face):
    """Face crop (grayscale or BGR, any size) -> (emotion, confidence proxy in [0, 1], time in ms).

    LinearSVC has no probabilities: the confidence is a softmax over the decision scores.
    """
    start = time.perf_counter()
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    scores = _model.decision_function(extract_features(gray).reshape(1, -1))[0]
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    i = int(np.argmax(probabilities))
    latency_ms = (time.perf_counter() - start) * 1000
    return str(_model.classes_[i]), float(probabilities[i]), latency_ms
