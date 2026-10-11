# 1  
import os

import cv2
import dlib
import joblib
from face_expression import detect_expression
 
# 2  
def plot_rectangle(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255,0,0), 4)
    return image
 
def main():
    # define facial expression
    dict_expression = {"angry" : 0 , "disgust" : 1 , "fear" : 2 , "happy" : 3 , "neutral" : 4 , "sad" : 5 , "surprise" : 6}

    #load model
    neigh = joblib.load('knn_model.pkl')

    # Construido UMA vez: refazer isso a cada quadro custa caro.
    detector = dlib.get_frontal_face_detector()

    # 3 open cam
    # No WSL nao existe /dev/video0. O webcam_server.py (na pasta de cima)
    # roda no Windows e publica a camera aqui. Para webcam local, volte
    # para cv2.VideoCapture(0).
    stream_url = os.environ.get("WEBCAM_URL", "http://localhost:8080/video")
    capture = cv2.VideoCapture(stream_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # sem isso o stream atrasa
    # 4 
    if capture.isOpened() is False:
        print("Camera Error ! Nao abri", stream_url)
        print("O webcam_server.py esta rodando no Windows?")
        return
    # 5 read frame
    while True:
        ret, frame = capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # BGR to GRAY
            gray = cv2.resize(gray, (int(1920/4), int(1080/4))) 
 
            # 6 use detector in dlib to extract face
            det_result = detector(gray, 1)  # detect and get result
            # As caixas vem do gray redimensionado; para desenhar no
            # frame original e preciso desfazer essa escala.
            scale_x = frame.shape[1] / gray.shape[1]
            scale_y = frame.shape[0] / gray.shape[0]

            for face in det_result:
                face_img = gray[ face.top() : face.bottom(), face.left():face.right()]
                if face.top()  > 0 :
                    face_img = cv2.resize(face_img, (48, 48)) 
                    cv2.imshow("entrada do classificador (48x48)", face_img)
                    
                    expression = detect_expression(face_img, neigh)
                    expression_str = list(dict_expression.keys())[list(dict_expression.values()).index(expression)]
                    print(expression_str)

                    left = int(face.left() * scale_x)
                    top = int(face.top() * scale_y)
                    right = int(face.right() * scale_x)
                    bottom = int(face.bottom() * scale_y)
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, expression_str, (left, max(top - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            








            # 8 mostra o quadro inteiro, em cores, com as caixas
            cv2.imshow("face detection with dlib", frame)
 
            # 9  "ESC"，to exit
            if cv2.waitKey(1) == 27:
                break
 
    # 10 release
    capture.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()