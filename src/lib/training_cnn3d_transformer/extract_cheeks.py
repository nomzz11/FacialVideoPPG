import torch
import numpy as np
import os
from PIL import Image
import cv2
from facenet_pytorch import MTCNN

# Initialisation du détecteur de visages MTCNN
mtcnn = MTCNN(
    select_largest=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Applique CLAHE (Correction de contraste adaptative) pour améliorer la détection faciale
    sur tout type de peau sans distorsion excessive.

    Args:
        image (numpy.ndarray): Image BGR (OpenCV)
        clip_limit (float): Intensité de la correction (plus élevé = plus de contraste)
        grid_size (tuple): Taille de la grille pour le traitement adaptatif

    Returns:
        numpy.ndarray: Image améliorée en BGR
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)  # Conversion BGR → LAB
    l, a, b = cv2.split(lab)  # Séparation des canaux

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)  # Appliquer CLAHE sur la luminance

    lab = cv2.merge((l, a, b))  # Fusionner les canaux
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Retour en BGR

    return Image.fromarray(enhanced_image)


def extract_cheeks(frame_pil, frame_id, output_size=112):
    """
    Détecte les joues dans une image et les renvoie sous forme d'une image carrée fusionnée.

    Args:
        frame_pil (PIL.Image): Image d'entrée au format PIL (RGB).
        output_size (int): Taille de sortie du carré final.

    Returns:
        np.ndarray ou None: Image fusionnée des joues sous forme de tableau numpy (RGB) ou None si échec.
    """

    frame_pil = apply_clahe(frame_pil)  # Appliquer avant MTCNN

    print(f"extract_cheeks appelée pour frame {frame_id}")

    # Détection des visages avec MTCNN
    boxes, probs, landmarks = mtcnn.detect(frame_pil, landmarks=True)

    if landmarks is None:
        print("Aucun visage détecté.")

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        NO_FACE_DIR = os.path.join(project_root, "stats")

        # Sauvegarde l'image si aucun visage détecté
        if frame_id is not None:
            save_path = os.path.join(NO_FACE_DIR, f"frame_{frame_id}.png")
        else:
            save_path = os.path.join(NO_FACE_DIR, "frame_unknown.png")

        frame_pil.save(save_path)
        print(f"Image sauvegardée : {save_path}")

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

    selected_landmarks = np.array(selected_landmarks)
    left_eye, right_eye, nose, mouth_left, mouth_right = selected_landmarks

    # Convertir l'image PIL en numpy pour OpenCV
    frame = np.array(frame_pil)

    # Définir les coordonnées des joues (approximatives)
    x1_l, x2_l = sorted([int(left_eye[0] - 40), int(mouth_left[0])])
    y1_l, y2_l = sorted([int(left_eye[1] + 40), int(mouth_left[1] + 40)])

    x1_r, x2_r = sorted([int(right_eye[0] + 40), int(mouth_right[0])])
    y1_r, y2_r = sorted([int(right_eye[1] + 40), int(mouth_right[1] + 40)])

    cheek_left = frame[y1_l:y2_l, x1_l:x2_l]
    cheek_right = frame[y1_r:y2_r, x1_r:x2_r]

    if cheek_left.size == 0 or cheek_right.size == 0:
        # Vérifier si les joues sont extraites
        print(f"left_cheek: {cheek_left}, right_cheek: {cheek_right}")
        print("Problème d'extraction des joues (taille 0).")
        return None

    # Redimensionner chaque joue pour garantir un carré
    cheek_left_resized = cv2.resize(cheek_left, (output_size // 2, output_size))
    cheek_right_resized = cv2.resize(cheek_right, (output_size // 2, output_size))

    # Fusionner les joues pour obtenir une image carrée
    combined_cheeks = np.hstack((cheek_left_resized, cheek_right_resized))

    return combined_cheeks
