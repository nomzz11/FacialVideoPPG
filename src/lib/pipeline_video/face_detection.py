# face_detection.py
import cv2
import numpy as np
from PIL import Image
from src.lib.training_cnn3d_transformer import extract_cheeks


def normalize_image(self, image):
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1)) + 1e-8  # Évite division par zéro
    normalized_image = (image - mean) / std
    return Image.fromarray((normalized_image * 255).astype(np.uint8))


def detect_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cheeks = extract_cheeks(frame_pil, frame_id)

        normalized_image = normalize_image(cheeks)

        if cheeks is not None:
            frames.append(normalized_image)
        frame_id += 1

    cap.release()
    return frames
