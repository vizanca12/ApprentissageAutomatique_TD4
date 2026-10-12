"""Web interface comparing the facial expression models side by side.

Each frame (webcam or uploaded image) goes through:
    face detection (dlib HOG) -> crop -> every model -> emotion, confidence, latency

    python app.py

then open the address printed by Gradio (usually http://localhost:7860).
"""
import os

import cv2
import dlib
import gradio as gr
import numpy as np

from cnn import cnn_model
from knn import knn_model

# Every model module exposes predict(face) -> (emotion, confidence, ms)
MODELS = {"LBP + KNN": knn_model}
try:
    from svm import svm_model
    MODELS["HOG + SVM"] = svm_model
except ImportError:
    pass  # SVM part not added yet
MODELS["mini-Xception (CNN)"] = cnn_model
SHORT_NAMES = {"LBP + KNN": "KNN", "HOG + SVM": "SVM", "mini-Xception (CNN)": "CNN"}

# The dlib HOG detector is the bottleneck, so the video is analysed at a
# small width and the boxes are scaled back. Uploads can afford more.
STREAM_WIDTH = 480
UPLOAD_WIDTH = 1024

BOX_COLOR = (37, 99, 235)     # RGB: Gradio frames are RGB, not BGR
TEXT_COLOR = (16, 185, 129)
TABLE_HEADERS = ["Model", "Emotion", "Confidence", "Latency (ms)"]

detector = dlib.get_frontal_face_detector()

# Warm-up: load the models and run them once, so the first frame is not slow
for model in MODELS.values():
    model.predict(np.zeros((48, 48), dtype=np.uint8))


def analyse(frame, upsample=0, detect_width=STREAM_WIDTH, auto_retry=False):
    """RGB frame -> (annotated frame, results table, status message)."""
    if frame is None:
        return None, [], "Upload an image to start."

    annotated = frame.copy()
    gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray_full.shape

    # Never upscale: only shrink when the image is wider than the target
    target = min(detect_width, width)
    scale = target / float(width)
    gray = cv2.resize(gray_full, (target, max(1, int(height * scale))))

    # With auto_retry, try larger upsampling to find small or distant faces
    attempts = [int(upsample)]
    if auto_retry:
        attempts += [u for u in (1, 2) if u > int(upsample)]
    faces = []
    for used_upsample in attempts:
        faces = detector(gray, used_upsample)
        if faces:
            break

    if not faces:
        return (annotated, [],
                "No face found. Try a more frontal, closer or better lit "
                "photo - the dlib HOG detector only finds frontal faces.")

    # Boxes back to full resolution, clipped to the image
    boxes = []
    for face in faces:
        left, top = max(int(face.left() / scale), 0), max(int(face.top() / scale), 0)
        right, bottom = min(int(face.right() / scale), width), min(int(face.bottom() / scale), height)
        if right - left >= 10 and bottom - top >= 10:
            boxes.append((left, top, right, bottom))
    if not boxes:
        return annotated, [], "Face too small."

    # The table shows the largest face; every face gets a box
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    table = []
    for k, (left, top, right, bottom) in enumerate(boxes):
        crop = gray_full[top:bottom, left:right]
        results = {name: model.predict(crop) for name, model in MODELS.items()}

        # One label line per model, above the box (or below it if there is no room).
        # Text size follows the image size so it stays readable on large photos.
        font = max(0.6, width / 1000)
        thick, step = max(2, int(font * 2)), int(35 * font)
        cv2.rectangle(annotated, (left, top), (right, bottom), BOX_COLOR, thick)
        y0 = top - 10 - step * (len(results) - 1)
        if y0 < step:
            y0 = bottom + step
        for line, (name, (emotion, _, _)) in enumerate(results.items()):
            cv2.putText(annotated, f"{SHORT_NAMES.get(name, name)}: {emotion}", (left, y0 + step * line),
                        cv2.FONT_HERSHEY_SIMPLEX, font, TEXT_COLOR, thick, cv2.LINE_AA)

        if k == 0:
            table = [[name, emotion, f"{conf:.0%}", round(ms, 1)]
                     for name, (emotion, conf, ms) in results.items()]

    status = f"{len(boxes)} face(s) found (upsample {used_upsample})."
    return annotated, table, status


def analyse_stream(frame, upsample):
    annotated, table, _ = analyse(frame, upsample, STREAM_WIDTH, auto_retry=False)
    return annotated, table


def analyse_upload(frame, upsample):
    return analyse(frame, upsample, UPLOAD_WIDTH, auto_retry=True)


DESCRIPTION = """
Face detection with **dlib** (HOG), then the same face crop goes through
every model: **LBP + KNN**, **HOG + SVM** and **mini-Xception (CNN)**.
The table shows the predicted emotion, confidence and latency of each model.
"""

with gr.Blocks(title="Facial Expression Recognition") as demo:
    gr.Markdown("# Facial Expression Recognition — model comparison")
    gr.Markdown(DESCRIPTION)

    upsample = gr.Slider(
        0, 2, value=0, step=1, label="Detector upsample",
        info="0 is fast and fits video. 1 or 2 find smaller or more distant "
             "faces, but are much slower.")

    with gr.Tabs():
        with gr.Tab("Live webcam"):
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
                cam_out = gr.Image(type="numpy", label="Detection")
            cam_table = gr.Dataframe(headers=TABLE_HEADERS, label="Predictions")

        with gr.Tab("Upload an image"):
            with gr.Row():
                upload = gr.Image(sources=["upload", "clipboard"], type="numpy", label="Image")
                up_out = gr.Image(type="numpy", label="Detection")
            up_table = gr.Dataframe(headers=TABLE_HEADERS, label="Predictions")
            up_status = gr.Markdown("Upload an image to start.")
            analyse_button = gr.Button("Analyse", variant="primary")

    cam.stream(fn=analyse_stream, inputs=[cam, upsample], outputs=[cam_out, cam_table],
               stream_every=0.2, concurrency_limit=1, show_progress="hidden")

    for event in (analyse_button.click, upload.change):
        event(fn=analyse_upload, inputs=[upload, upsample],
              outputs=[up_out, up_table, up_status])


if __name__ == "__main__":
    # Without a fixed port Gradio uses the first free one from 7860.
    # To force a port: set GRADIO_SERVER_PORT=7865
    port = os.environ.get("GRADIO_SERVER_PORT")
    # share=True would publish a public link on the internet - left off on purpose
    demo.launch(server_name="127.0.0.1", server_port=int(port) if port else None)
