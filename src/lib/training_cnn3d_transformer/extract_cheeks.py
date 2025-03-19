import torch
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import MTCNN

# Initialisation du détecteur de visages MTCNN
mtcnn = MTCNN(
    select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu"
)


def extract_cheeks(frame_pil, output_size=112):
    """
    Détecte les joues dans une image et les renvoie sous forme d'une image carrée fusionnée.

    Args:
        frame_pil (PIL.Image): Image d'entrée au format PIL (RGB).
        output_size (int): Taille de sortie du carré final.

    Returns:
        np.ndarray ou None: Image fusionnée des joues sous forme de tableau numpy (RGB) ou None si échec.
    """
    # Détection des visages avec MTCNN
    boxes, probs, landmarks = mtcnn.detect(frame_pil, landmarks=True)

    if landmarks is None:
        print("Aucun visage détecté.")
        return None

    # Récupération des landmarks
    landmark = landmarks[0]
    left_eye, right_eye, nose, mouth_left, mouth_right = landmark

    # Convertir l'image PIL en numpy pour OpenCV
    frame = np.array(frame_pil)

    # Définition des zones des joues
    cheek_left_x = int(landmark[0][0] - 0.4 * abs(landmark[0][0] - landmark[3][0]))
    cheek_left = frame[
        int(landmark[0][1]) : int(
            landmark[3][1]
        ),  # Hauteur : de l'œil gauche à la bouche gauche
        cheek_left_x : int(landmark[3][0]),  # Largeur : en s'étendant vers la gauche
    ]

    cheek_right_x = int(landmark[4][0] + 0.4 * abs(landmark[1][0] - landmark[4][0]))
    cheek_right = frame[
        int(landmark[1][1]) : int(
            landmark[4][1]
        ),  # Hauteur : de l'œil droit à la bouche droite
        int(landmark[4][0]) : cheek_right_x,  # Largeur : en s'étendant vers la droite
    ]

    # Vérification des dimensions des joues
    if cheek_left.size == 0 or cheek_right.size == 0:
        print("Problème d'extraction des joues (taille 0).")
        return None

    # **Redimensionner chaque joue en rectangle (112, 56)**
    cheek_left_resized = cv2.resize(cheek_left, (56, 112))
    cheek_right_resized = cv2.resize(cheek_right, (56, 112))

    # **Fusionner les joues en une image carrée (112, 112)**
    combined_cheeks = np.hstack((cheek_left_resized, cheek_right_resized))

    return combined_cheeks  # Image finale carrée sous forme de np.ndarray
