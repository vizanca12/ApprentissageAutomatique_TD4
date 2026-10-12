"""Interface web do reconhecimento de expressoes faciais.

A webcam e capturada pelo NAVEGADOR, no lado do Windows, e os quadros
chegam aqui como arrays numpy. Por isso este app nao precisa do
webcam_server.py nem enxergar /dev/video0.

    source .venv/bin/activate
    python app.py

e abra o endereco que o Gradio imprimir (normalmente
http://localhost:7860) no navegador do Windows.
"""

import os

import cv2
import dlib
import gradio as gr
import joblib
import numpy as np

from face_expression import expression_features

HERE = os.path.dirname(os.path.abspath(__file__))

# A ordem segue model.classes_ == [0..6], o mesmo dict_expression do main.py
EXPRESSIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Largura usada na deteccao. O detector HOG do dlib e o gargalo, entao no
# video detectamos pequeno e reescalamos as caixas de volta. No upload nao
# ha pressa, entao usamos mais resolucao.
STREAM_WIDTH = 480
UPLOAD_WIDTH = 1024

BOX_COLOR = (37, 99, 235)    # RGB - os quadros do Gradio vem em RGB, nao BGR
TEXT_COLOR = (16, 185, 129)

model = joblib.load(os.path.join(HERE, "knn_model.pkl"))
detector = dlib.get_frontal_face_detector()


def detect_faces(gray, upsample):
    return detector(gray, int(upsample))


def analyse(frame, upsample=0, detect_width=STREAM_WIDTH, auto_retry=False):
    """Recebe um quadro RGB.

    Devolve (quadro anotado, probabilidades, mensagem de status).
    Com auto_retry, se nada for encontrado no upsample pedido, tenta os
    maiores antes de desistir - rostos pequenos ou distantes so aparecem
    assim, e no upload o custo extra nao incomoda.
    """
    if frame is None:
        return None, {}, "Envie uma imagem para comecar."

    annotated = frame.copy()
    gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray_full.shape

    # Nunca ampliar: so reduzimos quando a imagem e maior que o alvo.
    target = min(detect_width, width)
    scale = target / float(width)
    gray = cv2.resize(gray_full, (target, max(1, int(height * scale))))

    attempts = [int(upsample)]
    if auto_retry:
        attempts += [u for u in (1, 2) if u > int(upsample)]

    faces = []
    used_upsample = int(upsample)
    for attempt in attempts:
        faces = detect_faces(gray, attempt)
        used_upsample = attempt
        if faces:
            break

    if not faces:
        return (annotated, {},
                "Nenhum rosto encontrado. Tente uma foto mais frontal, mais "
                "proxima ou com melhor iluminacao - o detector HOG do dlib "
                "so pega rostos de frente.")

    probabilities = {}
    for face in faces:
        # dlib pode devolver caixas que saem da imagem; cortar antes do crop.
        top = max(face.top(), 0)
        left = max(face.left(), 0)
        bottom = min(face.bottom(), gray.shape[0])
        right = min(face.right(), gray.shape[1])
        if bottom - top < 10 or right - left < 10:
            continue

        crop = cv2.resize(gray[top:bottom, left:right], (48, 48))
        scores = model.predict_proba(np.array([expression_features(crop)]))[0]
        label = EXPRESSIONS[int(np.argmax(scores))]

        # O rosto mais confiante alimenta o painel de probabilidades.
        if not probabilities or max(scores) > max(probabilities.values()):
            probabilities = {name: float(scores[i])
                             for i, name in enumerate(EXPRESSIONS)}

        box = [int(v / scale) for v in (left, top, right, bottom)]
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]),
                      BOX_COLOR, 2)
        cv2.putText(annotated, label, (box[0], max(box[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2,
                    cv2.LINE_AA)

    status = "{} rosto(s) encontrado(s) com upsample {}.".format(
        len(faces), used_upsample)
    return annotated, probabilities, status


def analyse_stream(frame, upsample):
    annotated, probabilities, _ = analyse(
        frame, upsample, detect_width=STREAM_WIDTH, auto_retry=False)
    return annotated, probabilities


def analyse_upload(frame, upsample):
    return analyse(frame, upsample, detect_width=UPLOAD_WIDTH, auto_retry=True)


DESCRIPTION = """
Deteccao de rosto com **dlib** (HOG) e classificacao da expressao com
**KNN sobre histogramas de LBP** (Local Binary Patterns), 160 features
por rosto.

A webcam e lida pelo seu navegador, entao isto funciona mesmo com o
Python rodando no WSL.
"""

with gr.Blocks(title="Reconhecimento de Expressoes Faciais") as demo:
    gr.Markdown("# Reconhecimento de Expressoes Faciais")
    gr.Markdown(DESCRIPTION)

    upsample = gr.Slider(
        0, 2, value=0, step=1, label="Upsample do detector",
        info="0 e rapido e serve para video. 1 ou 2 acham rostos menores "
             "ou mais distantes, porem ficam bem mais lentos.")

    # gr.Tabs explicito: sem ele o Gradio montava um segundo cabecalho de
    # abas que nao trocava o conteudo ao ser clicado.
    with gr.Tabs():
        with gr.Tab("Webcam ao vivo"):
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True,
                               type="numpy", label="Webcam")
                cam_out = gr.Image(type="numpy", label="Deteccao")
                cam_scores = gr.Label(num_top_classes=3, label="Expressao")

        with gr.Tab("Enviar uma imagem"):
            with gr.Row():
                upload = gr.Image(sources=["upload", "clipboard"],
                                  type="numpy", label="Imagem")
                up_out = gr.Image(type="numpy", label="Deteccao")
                up_scores = gr.Label(num_top_classes=7, label="Expressao")
            up_status = gr.Markdown("Envie uma imagem para comecar.")
            analyse_button = gr.Button("Analisar", variant="primary")

    cam.stream(fn=analyse_stream, inputs=[cam, upsample],
               outputs=[cam_out, cam_scores],
               stream_every=0.2, concurrency_limit=1, show_progress="hidden")

    for event in (analyse_button.click, upload.change):
        event(fn=analyse_upload, inputs=[upload, upsample],
              outputs=[up_out, up_scores, up_status])


if __name__ == "__main__":
    # Sem server_port fixo o Gradio varre 7860-7959 e usa a primeira porta
    # livre, em vez de morrer se sobrou uma instancia antiga. Para forcar
    # uma porta: GRADIO_SERVER_PORT=7865 python app.py
    port = os.environ.get("GRADIO_SERVER_PORT")

    # share=True publicaria um link publico na internet - deixado desligado
    # de proposito. Ligue so se quiser mostrar para alguem de fora.
    demo.launch(server_name="127.0.0.1",
                server_port=int(port) if port else None)
