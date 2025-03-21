import cv2
import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Chemins des dossiers
INPUT_DIR = "refined_data"  # Dossier contenant les 202 dossiers vidéo avec frames
OUTPUT_DIR = "data_mtcnn"  # Dossier où seront enregistrées les joues extraites

# Initialisation du détecteur de visages MTCNN
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initialisation de MTCNN sur {device}")
mtcnn = MTCNN(select_largest=True, device=device)


def normalize_image(image, normalization_type="z-score"):
    """
    Normalise une image selon différentes méthodes.

    Args:
        image: Image à normaliser (numpy array)
        normalization_type: Type de normalisation ('z-score', 'minmax', 'clahe', 'histogram')

    Returns:
        Image normalisée
    """
    if normalization_type == "z-score":
        # Z-score normalization (standardisation)
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1)) + 1e-8  # Évite division par zéro
        normalized = (image - mean) / std
        # Ramener à l'échelle 0-255 pour la visualisation
        normalized = np.clip(normalized * 64 + 128, 0, 255).astype(np.uint8)

    elif normalization_type == "minmax":
        # Min-Max normalization
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

    elif normalization_type == "clahe":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:  # Image RGB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:  # Image en niveaux de gris
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image)

    elif normalization_type == "histogram":
        # Égalisation d'histogramme
        if len(image.shape) == 3:  # Image RGB
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge((h, s, v))
            normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:  # Image en niveaux de gris
            normalized = cv2.equalizeHist(image)

    else:
        # Pas de normalisation
        normalized = image.copy()

    return normalized


def apply_multiple_normalizations(image):
    """
    Applique plusieurs types de normalisation et retourne l'image optimisée.
    Cette approche combine les avantages de plusieurs méthodes.
    """
    # 1. Appliquer CLAHE pour améliorer le contraste local
    clahe_result = normalize_image(image, "clahe")

    # 2. Standardiser l'image (pour réduire l'effet des conditions d'éclairage)
    normalized = normalize_image(clahe_result, "z-score")

    return normalized


def extract_cheeks(frame_pil, frame_id, video_folder, output_size=112):
    """
    Détecte les joues dans une image et les renvoie sous forme d'une image carrée fusionnée.
    """
    try:
        # Détection des visages avec MTCNN
        boxes, probs, landmarks = mtcnn.detect(frame_pil, landmarks=True)

        if landmarks is None or boxes is None:
            return None

        # Sélection du visage le plus grand si plusieurs sont détectés
        if landmarks is not None and len(landmarks) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_index = np.argmax(areas)
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

        # Assurer que les coordonnées sont dans les limites de l'image
        height, width = frame.shape[:2]
        x1_l, x2_l = max(0, x1_l), min(width, x2_l)
        y1_l, y2_l = max(0, y1_l), min(height, y2_l)
        x1_r, x2_r = max(0, x1_r), min(width, x2_r)
        y1_r, y2_r = max(0, y1_r), min(height, y2_r)

        # Vérifier que les dimensions sont valides
        if x1_l >= x2_l or y1_l >= y2_l or x1_r >= x2_r or y1_r >= y2_r:
            return None

        cheek_left = frame[y1_l:y2_l, x1_l:x2_l]
        cheek_right = frame[y1_r:y2_r, x1_r:x2_r]

        if cheek_left.size == 0 or cheek_right.size == 0:
            return None

        # Normaliser chaque joue individuellement avant redimensionnement
        cheek_left_norm = apply_multiple_normalizations(cheek_left)
        cheek_right_norm = apply_multiple_normalizations(cheek_right)

        # Redimensionner chaque joue pour garantir un carré
        cheek_left_resized = cv2.resize(
            cheek_left_norm, (output_size // 2, output_size)
        )
        cheek_right_resized = cv2.resize(
            cheek_right_norm, (output_size // 2, output_size)
        )

        # Fusionner les joues pour obtenir une image carrée
        combined_cheeks = np.hstack((cheek_left_resized, cheek_right_resized))

        # Normalisation finale pour l'ensemble de l'image (optionnel)
        # combined_cheeks = apply_multiple_normalizations(combined_cheeks)

        return combined_cheeks

    except Exception as e:
        print(
            f"Erreur lors de l'extraction des joues pour {frame_id} de {video_folder}: {e}"
        )
        return None


def process_video_folder(video_folder, output_size=112):
    """
    Traite toutes les frames d'un dossier vidéo spécifique.

    Args:
        video_folder (str): Nom du dossier vidéo à traiter.
        output_size (int): Taille de sortie des joues fusionnées.
    """
    input_path = os.path.join(INPUT_DIR, video_folder)
    output_path = os.path.join(OUTPUT_DIR, video_folder)

    # Création du dossier de sortie
    os.makedirs(output_path, exist_ok=True)

    # Liste des frames dans le dossier d'entrée
    frames = [
        f for f in os.listdir(input_path) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    frames.sort()  # Trier les frames par ordre alphabétique

    success_count = 0
    fail_count = 0

    print(f"Traitement de {video_folder}: {len(frames)} frames trouvées")

    # Traiter chaque frame
    for frame_file in tqdm(frames, desc=f"Traitement de {video_folder}"):
        frame_path = os.path.join(input_path, frame_file)
        output_file = os.path.join(output_path, frame_file)

        # Vérifier si le fichier existe déjà (pour reprendre un traitement interrompu)
        if os.path.exists(output_file):
            success_count += 1
            continue

        try:
            # Charger l'image
            frame_pil = Image.open(frame_path).convert("RGB")

            # Extraire les joues
            cheeks = extract_cheeks(frame_pil, frame_file, video_folder, output_size)

            if cheeks is not None:
                # Sauvegarder l'image des joues
                cv2.imwrite(output_file, cv2.cvtColor(cheeks, cv2.COLOR_RGB2BGR))
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"Erreur lors du traitement de {frame_path}: {e}")
            fail_count += 1

    return video_folder, success_count, fail_count, len(frames)


def process_all_videos(output_size=112):
    """
    Traite tous les dossiers vidéo dans le répertoire d'entrée.
    """
    # Création du dossier de sortie principal
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Liste des dossiers vidéo
    video_folders = [
        d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))
    ]
    video_folders.sort()

    print(f"Début du traitement de {len(video_folders)} dossiers vidéo")

    results = []
    for video_folder in video_folders:
        result = process_video_folder(video_folder, output_size)
        results.append(result)

        # Afficher le résultat après chaque vidéo
        video, success, fail, total = result
        print(f"Vidéo {video}: {success} réussites, {fail} échecs, {total} total")

    # Afficher un résumé
    total_success = sum(r[1] for r in results)
    total_fail = sum(r[2] for r in results)
    total_frames = sum(r[3] for r in results)

    print("\n=== RÉSUMÉ ===")
    print(f"Total: {total_success + total_fail}/{total_frames} frames traitées")
    print(f"Réussites: {total_success} ({total_success/total_frames*100:.1f}%)")
    print(f"Échecs: {total_fail} ({total_fail/total_frames*100:.1f}%)")

    # Afficher les vidéos avec moins de 50% de réussite
    low_success = [(r[0], r[1], r[3]) for r in results if r[1] < r[3] * 0.5]
    if low_success:
        print("\nVidéos avec moins de 50% de réussite:")
        for video, success, total in low_success:
            print(f"  - {video}: {success}/{total} ({success/total*100:.1f}%)")


if __name__ == "__main__":
    print("Programme d'extraction des joues à partir des frames vidéo")
    print(f"Dossier d'entrée: {INPUT_DIR}")
    print(f"Dossier de sortie: {OUTPUT_DIR}")
    process_all_videos(output_size=112)
