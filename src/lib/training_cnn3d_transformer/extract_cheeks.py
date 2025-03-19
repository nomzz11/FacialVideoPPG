import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(
    select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu"
)


def extract_cheeks_square(frame, output_size=112):
    """
    Détecte précisément les joues (gauche et droite), les combine en une image carrée.

    Retourne : une image carrée (RGB) prête à être utilisée par le modèle.
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

    if landmarks is None:
        print("Aucun visage détecté !")
        return None

    landmark = landmarks[0]

    left_eye, right_eye, nose, mouth_left, mouth_right = landmark

    # Joue Gauche
    cheek_left_x = int(
        landmarks[0][0][0] - 0.4 * abs(landmarks[0][0][0] - landmarks[0][3][0])
    )
    cheek_left = frame[
        int(landmarks[0][0][1]) : int(landmarks[0][3][1]),
        cheek_left_x : int(landmarks[0][3][0]),
    ]

    # Joue droite
    cheek_right_x = int(
        landmarks[0][4][0] + 0.4 * abs(landmarks[0][1][0] - landmarks[0][4][0])
    )
    cheek_right = frame[
        int(landmarks[0][1][1]) : int(landmarks[0][4][1]),
        int(landmarks[0][4][0]) : cheek_right_x,
    ]

    # Vérifier si les joues sont correctement extraites :
    if cheek_left.size == 0 or cheek_right.size == 0:
        print("Problème d'extraction des joues.")
        return None

    # Redimensionner clairement à la même taille pour chaque joue
    cheek_left_resized = cv2.resize(cheek_left, (112, 112))
    cheek_right_resized = cv2.resize(cheek_right, (112, 112))

    # Fusionner les joues en une image carrée claire :
    combined_cheeks = np.hstack((cheek_left, cheek_right))
    combined_cheeks = cv2.resize(combined_cheeks, (112, 112))

    return combined_cheeks
