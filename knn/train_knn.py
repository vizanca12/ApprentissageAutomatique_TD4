"""Train the LBP + KNN model on FER-2013 and evaluate it on the test set.

Run:  python knn/train_knn.py
"""
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from knn_model import MODEL_PATH, extract_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_features(split):
    X, y = [], []
    for emo_dir in sorted((DATA_DIR / split).iterdir()):   # sorted: same order on every OS
        for f in emo_dir.glob("*.jpg"):
            X.append(extract_features(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)))
            y.append(emo_dir.name)                          # label = emotion name, not an index
    return np.array(X, dtype=np.uint8), np.array(y)        # counts <= 144, fit in uint8


t0 = time.perf_counter()
X_train, y_train = load_features("train")
X_test, y_test = load_features("test")
print(f"Features: train {X_train.shape}, test {X_test.shape} ({time.perf_counter() - t0:.0f} s)")

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

joblib.dump(knn, MODEL_PATH, compress=3)
print(f"Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
