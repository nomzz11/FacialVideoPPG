import cv2
import numpy as np
from PIL import Image


def face_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    extracted_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
        )

        if len(faces) > 0:
            # Prendre le plus grand visage détecté
            x, y, w, h = sorted(
                faces, key=lambda rect: rect[2] * rect[3], reverse=True
            )[0]
            face_crop = frame[y : y + h, x : x + w]
            extracted_faces.append(
                Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            )

    cap.release()
    return extracted_faces  # Liste d'images de visages
