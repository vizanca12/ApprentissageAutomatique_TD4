"""LBP + KNN model for facial expression recognition (FER-2013).

Usage:
    import knn_model
    emotion, confidence, ms = knn_model.predict(face)

LBP histogram features adapted from the course examples by J. J. Garcia Cardenas.
The model is trained by train_knn.py.
"""
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern

# LBP parameters
P = 25                  # number of neighbors
R = 5                   # radius
METHOD = "uniform"
N_BINS = P + 2          # "uniform" LBP gives codes 0..P+1
GRID = 4                # face split into 4x4 regions
SIZE = 48               # FER-2013 image size

MODEL_PATH = Path(__file__).parent / "knn_model.pkl"
_model = None


def preprocess(face):
    """Face crop (grayscale or BGR, any size) -> 48x48 grayscale."""
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cv2.resize(face, (SIZE, SIZE))


def extract_features(face):
    """48x48 grayscale face -> concatenated LBP histograms (GRID*GRID*N_BINS values)."""
    lbp = local_binary_pattern(face, P, R, method=METHOD)
    step = SIZE // GRID
    hists = [np.histogram(lbp[i:i + step, j:j + step], bins=N_BINS, range=(0, N_BINS))[0]
             for i in range(0, SIZE, step) for j in range(0, SIZE, step)]
    return np.concatenate(hists)


def predict(face):
    """Face crop -> (emotion, confidence in [0, 1], time in ms)."""
    global _model
    if _model is None:  # loaded on the first call, so train_knn.py can import this file
        _model = joblib.load(MODEL_PATH)
    t0 = time.perf_counter()
    probs = _model.predict_proba([extract_features(preprocess(face))])[0]
    i = int(np.argmax(probs))
    ms = (time.perf_counter() - t0) * 1000
    return str(_model.classes_[i]), float(probs[i]), ms
