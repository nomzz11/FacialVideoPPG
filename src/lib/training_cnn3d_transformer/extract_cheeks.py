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

    if landmarks is not None and len(landmarks) > 1:
        # Calculer la surface de chaque boîte englobante (hauteur * largeur)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Trouver l'index du visage ayant la plus grande surface
        largest_index = np.argmax(areas)

        # Sélectionner les landmarks du plus grand visage
        selected_landmarks = landmarks[largest_index]
    else:
        selected_landmarks = landmarks[0] if landmarks is not None else None
    # Récupération des landmarks
    landmark = landmarks[0]
    print("landmarks : ", landmarks)
    print("landmark", selected_landmarks)
    left_eye, right_eye, nose, mouth_left, mouth_right = selected_landmarks

    # Convertir l'image PIL en numpy pour OpenCV
    frame = np.array(frame_pil)

    # Définir les coordonnées des joues (approximatives)
    cheek_left_top = (int(left_eye[0] - 40), int(left_eye[1] + 40))
    cheek_left_bottom = (int(mouth_left[0]), int(mouth_left[1] + 40))

    cheek_right_top = (int(right_eye[0] + 40), int(right_eye[1] + 40))
    cheek_right_bottom = (int(mouth_right[0]), int(mouth_right[1] + 40))

    # Extraire les joues
    cheek_left = frame[
        cheek_left_top[1] : cheek_left_bottom[1],
        cheek_left_top[0] : cheek_left_bottom[0],
    ]
    cheek_right = frame[
        cheek_right_top[1] : cheek_right_bottom[1],
        cheek_right_top[0] : cheek_right_bottom[0],
    ]

    # Vérifier si les joues sont extraites
    if cheek_left.size == 0 or cheek_right.size == 0:
        print("Problème d'extraction des joues (taille 0).")
        return None

    # Redimensionner chaque joue pour garantir un carré
    cheek_left_resized = cv2.resize(cheek_left, (output_size // 2, output_size))
    cheek_right_resized = cv2.resize(cheek_right, (output_size // 2, output_size))

    # Fusionner les joues pour obtenir une image carrée
    combined_cheeks = np.hstack((cheek_left_resized, cheek_right_resized))

    return combined_cheeks
