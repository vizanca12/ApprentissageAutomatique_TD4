"""mini-Xception CNN for facial expression recognition (pretrained on FER-2013).

Usage:
    import cnn_model
    emotion, confidence, ms = cnn_model.predict(face)
"""
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # FER-2013 order

# Load the model once, when the module is imported
_MODEL_PATH = Path(__file__).parent / "models" / "fer2013_mini_XCEPTION.102-0.66.hdf5"
_model = tf.keras.models.load_model(_MODEL_PATH, compile=False)
_SIZE = _model.input_shape[1]  # 64


def preprocess(face):
    """Face crop (grayscale or BGR, any size) -> array (1, 64, 64, 1) scaled to [-1, 1]."""
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (_SIZE, _SIZE)).astype("float32") / 255.0
    return ((face - 0.5) * 2.0).reshape(1, _SIZE, _SIZE, 1)


def predict(face):
    """Face crop -> (emotion, confidence in [0, 1], time in ms)."""
    t0 = time.perf_counter()
    probs = _model([preprocess(face)], training=False).numpy()[0]
    i = int(np.argmax(probs))
    ms = (time.perf_counter() - t0) * 1000
    return EMOTIONS[i], float(probs[i]), ms
