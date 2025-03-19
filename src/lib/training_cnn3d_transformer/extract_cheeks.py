import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(
    select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu"
)


def extract_cheeks(frame, output_size=112):
    """
    Détecte précisément les joues gauche et droite depuis une frame avec MTCNN.
    Retourne une image carrée combinant les deux joues.
    """
    img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if landmarks is None or len(landmarks) == 0:
        print("Aucun visage détecté")
        return None

    landmarks = landmarks[0]

    # Landmarks précis pour joues
    left_eye, right_eye, _, mouth_left, mouth_right = landmarks

    h, w, _ = frame.shape

    # Joues Gauche
    cheek_left_x1 = int(max(mouth_left[0] - 0.6 * abs(mouth_left[0] - left_eye[0]), 0))
    cheek_left_x2 = int(mouth_left[0])
    cheek_left_y1 = int(left_eye[1])
    cheek_left_y2 = int(mouth_left[1] + 0.2 * abs(mouth_left[1] - left_eye[1]))
    cheek_left_y2 = min(cheek_left_y2, h)

    left_cheek = frame[cheek_left_y1:cheek_left_y2, cheek_left_x1:cheek_left_x2]

    # Joue Droite
    cheek_right_x1 = int(mouth_right[0])
    cheek_right_x2 = int(
        min(mouth_right[0] + 0.6 * abs(right_eye[0] - mouth_right[0]), w)
    )
    cheek_right_y1 = int(right_eye[1])
    cheek_right_y2 = int(mouth_right[1] + 0.2 * abs(mouth_right[1] - right_eye[1]))
    cheek_right_y2 = min(cheek_right_y2, h)

    right_cheek = frame[cheek_right_y1:cheek_right_y2, cheek_right_x1:cheek_right_x2]

    # Vérification de la taille des joues
    if left_cheek.size == 0 or right_cheek.size == 0:
        print("Erreur extraction joues, vérifiez la détection des landmarks.")
        return None

    # Redimensionnement des joues
    left_cheek_resized = cv2.resize(left_cheek, (output_size, output_size))
    right_cheek_resized = cv2.resize(right_cheek, (output_size, output_size))

    # Combinaison claire des joues côte-à-côte
    combined_cheeks = np.hstack((left_cheek_resized, right_cheek_resized))

    # Redimension finale pour un carré parfait
    combined_cheeks_square = cv2.resize(combined_cheeks, (output_size, output_size))

    return combined_cheeks_square
