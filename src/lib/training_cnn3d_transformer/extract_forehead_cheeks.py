import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu"
)


def extract_forehead_cheeks_ordered(frame):
    image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs, landmarks = mtcnn.detect(image_rgb, landmarks=True)

    if boxes is None:
        print("Aucun visage détecté")
        return None

    landmark = landmarks[0]

    left_eye, right_eye, nose, mouth_left, mouth_right = landmark

    # FRONT
    forehead_top = int(
        min(left_eye[1], right_eye[1]) - 0.6 * abs(nose[1] - left_eye[1])
    )
    forehead_bottom = int(min(left_eye[1], right_eye[1]))
    forehead_left = int(left_eye[0])
    forehead_right = int(right_eye[0])
    forehead = frame[forehead_top:forehead_bottom, forehead_left:forehead_right]

    # JOUE GAUCHE
    cheek_left_top = int(left_eye[1])
    cheek_left_bottom = int(mouth_left[1])
    cheek_left_left = int(left_eye[0] - 0.4 * abs(left_eye[0] - mouth_left[0]))
    cheek_left_right = int(mouth_left[0])
    cheek_left = frame[
        cheek_left_top:cheek_left_bottom, cheek_left_left:cheek_left_right
    ]

    # JOUE DROITE
    cheek_right_top = int(right_eye[1])
    cheek_right_bottom = int(mouth_right[1])
    cheek_right_left = int(mouth_right[0])
    cheek_right_right = int(right_eye[0] + 0.4 * abs(right_eye[0] - mouth_right[0]))
    cheek_right = frame[
        cheek_right_top:cheek_right_bottom, cheek_right_left:cheek_right_right
    ]

    # Redimensionner les régions à la même taille
    size = (100, 100)  # exemple de taille fixe

    forehead = cv2.resize(forehead, size)
    cheek_left = cv2.resize(cheek_left, size)
    cheek_right = cv2.resize(cheek_right, size)

    # Créer clairement une image composite ordonnée :
    top_row = forehead
    bottom_row = np.hstack((cheek_left, cheek_right))
    composite_image = np.vstack((top_row, bottom_row))

    return composite_image
