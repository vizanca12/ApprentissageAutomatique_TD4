# Facial Expression Recognition — Model Comparison

ROB311 – Lab 4 (Prof. Adriana Tapus, Juan José García Cárdenas) · ENSTA Paris

A web interface that takes a webcam frame or an image, detects the face and compares
three facial expression classifiers side by side, with their prediction, confidence
and latency:

1. **LBP + KNN** — Local Binary Patterns histograms + k-nearest neighbors (baseline)
2. **HOG + Linear SVM** — Histogram of Oriented Gradients + one-vs-rest linear SVM
3. **mini-Xception (CNN)** — lightweight convolutional neural network

All models are trained/evaluated on **FER-2013** (48×48 grayscale faces, 7 emotions:
angry, disgust, fear, happy, neutral, sad, surprise).

## Results

Evaluated on the FER-2013 test set (7,178 images). Latency = mean time per face,
including preprocessing and feature extraction, on CPU.

| Model | Features | Test accuracy | Latency |
|---|---|---|---|
| LBP + KNN (k = 5) | LBP uniform (P = 25, R = 5), 4×4 regions → 432 values | 38.5% | ~47 ms |
| HOG + Linear SVM | HOG (9 orientations, 8×8 cells, 2×2 blocks) → 900 values | 43.3% | < 1 ms |
| mini-Xception (CNN) | learned by the network (64×64 input) | **65.5%** | ~27 ms |

- The CNN is clearly the most accurate: it learns its own features instead of using
  hand-crafted ones.
- The SVM is the fastest: a prediction is a single linear computation.
- The KNN is the slowest: each prediction compares the face with the 28,709 training examples.
- *Surprise* is among the easiest emotions for all three models. The SVM and the CNN
  do best on *happy* and worst on *fear* and *sad*, which are often confused with each
  other and with *neutral*.
- The SVM confidence is a softmax over the decision scores (LinearSVC has no
  probabilities), so it is only an approximation.

## Project structure

```
app.py            Gradio interface (face detection + the 3 models)
knn/              LBP + KNN
  knn_model.py      predict(face) used by the interface
  train_knn.py      training and evaluation script
  knn_model.pkl     trained model
  TP.ipynb          training and evaluation notebook
svm/              HOG + Linear SVM
  svm_model.py      predict(face) used by the interface
  hog_linear_svm.joblib
  TP_HOG.ipynb      training and evaluation notebook
cnn/              mini-Xception CNN
  cnn_model.py      predict(face) used by the interface
  cnn.ipynb         loading, evaluation and latency notebook
  models/           pretrained weights
data/             FER-2013 dataset (not included, see below)
```

Each `*_model.py` exposes the same function, so the interface treats the models identically:

```python
emotion, confidence, ms = predict(face)   # face: cropped face, grayscale or BGR, any size
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

**Dataset:** download FER-2013 from Kaggle
([msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)) and extract it so that
the folders are `data/train/<emotion>/` and `data/test/<emotion>/`.
The dataset is only needed to train or evaluate the models, not to run the interface.

## Usage

Run the interface, then open the address it prints (usually http://127.0.0.1:7860):

```bash
python app.py
```

- **Live webcam** tab: real-time predictions of the 3 models.
- **Upload an image** tab: analyse a photo.

Retrain the KNN (a few minutes; the notebooks `knn/TP.ipynb` and `svm/TP_HOG.ipynb` also retrain their model):

```bash
python knn/train_knn.py
```

## Credits

- mini-Xception and its pretrained weights: [oarriaga/face_classification](https://github.com/oarriaga/face_classification)
  (Arriaga, Valdenegro-Toro & Plöger, *Real-time Convolutional Neural Networks for Emotion and Gender Classification*, 2017).
- LBP histogram features adapted from the course examples by J. J. García Cárdenas.
- Dataset: FER-2013 (Goodfellow et al., *Challenges in Representation Learning*, ICML 2013 workshop).
- Face detection: [dlib](http://dlib.net/) HOG frontal face detector.

## Team

- João Victor Ferro Amorosini
- Luiza Gonçalves Soares
- Vinicius César Cavallaro Zancheta 
