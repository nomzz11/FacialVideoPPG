import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import heartpy as hp
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from torch.nn import Transformer
import torch.nn as nn
from torchvision.models.video import r3d_18


# TODO: SPLIT FUNCTIONS IN DIFFERENT FOLDERS

# Configuration
BATCH_SIZE = 16  # Ajustez selon votre mémoire GPU


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out, out  # Retourne aussi la carte d'attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_max_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(avg_max_out)
        attn_sigmoid = self.sigmoid(attn)
        return x * attn_sigmoid, attn_sigmoid


class CBAM3D(nn.Module):
    def __init__(self, in_planes):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x, ca_attention = self.ca(x)
        x, sa_attention = self.sa(x)
        return x, sa_attention


class r3d_transformer(nn.Module):
    def __init__(self, out_features=60):
        super(r3d_transformer, self).__init__()
        # Chargement du backbone R3D-18 pré-entraîné
        r3d = r3d_18(pretrained=True)

        # Extraction des couches individuelles
        self.stem = r3d.stem
        self.layer1 = r3d.layer1  # Gardons les layers séparés
        self.layer2 = r3d.layer2
        self.layer3 = r3d.layer3
        self.layer4 = r3d.layer4
        self.avgpool = r3d.avgpool

        # Ajout des modules CBAM3D
        self.cbam1 = CBAM3D(64)  # 64 canaux après layer1
        self.cbam2 = CBAM3D(128)  # 128 canaux après layer2

        # Couche finale
        self.fc = nn.Linear(512, out_features)

    def forward(self, x, seq_len=None):
        attention_maps = {}

        # Passage à travers le stem
        x = self.stem(x)

        # Layer 1 + CBAM
        x = self.layer1(x)
        x, attn1 = self.cbam1(x)
        attention_maps["layer1"] = attn1

        # Layer 2 + CBAM
        x = self.layer2(x)
        x, attn2 = self.cbam2(x)
        attention_maps["layer2"] = attn2

        # Couches restantes sans attention
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification finale
        x = self.fc(x)

        return x, attention_maps


# Initialisation du détecteur de visages MTCNN
mtcnn = MTCNN(
    select_largest=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def extract_cheeks(frame, frame_id, video_path, output_size=112):
    """
    Détecte les joues dans une image et les renvoie sous forme d'une image carrée fusionnée.
    """
    try:
        # Détection des visages avec MTCNN
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

        if landmarks is None or len(landmarks) == 0:
            print(f"Aucun visage détecté dans la frame {frame_id} de {video_path}")
            return None

        # Sélectionne le visage avec la plus grande surface si plusieurs sont détectés
        if len(landmarks) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_index = np.argmax(areas)
            selected_landmarks = landmarks[largest_index]
        else:
            selected_landmarks = landmarks[0]

        # Extraction des points clés du visage
        left_eye, right_eye, nose, mouth_left, mouth_right = selected_landmarks

        # Convertir l'image PIL en numpy pour OpenCV
        frame_np = np.array(frame)

        # Définir les coordonnées des joues (approximatives)
        x1_l, x2_l = sorted([int(left_eye[0] - 40), int(mouth_left[0])])
        y1_l, y2_l = sorted([int(left_eye[1] + 40), int(mouth_left[1] + 40)])

        x1_r, x2_r = sorted([int(right_eye[0] + 40), int(mouth_right[0])])
        y1_r, y2_r = sorted([int(right_eye[1] + 40), int(mouth_right[1] + 40)])

        # Vérification des coordonnées pour éviter les indices négatifs ou hors limites
        height, width = frame_np.shape[:2]
        x1_l, y1_l = max(0, x1_l), max(0, y1_l)
        x2_l, y2_l = min(width, x2_l), min(height, y2_l)
        x1_r, y1_r = max(0, x1_r), max(0, y1_r)
        x2_r, y2_r = min(width, x2_r), min(height, y2_r)

        # Extraction des régions des joues
        cheek_left = frame_np[y1_l:y2_l, x1_l:x2_l]
        cheek_right = frame_np[y1_r:y2_r, x1_r:x2_r]

        # Vérification que les régions ne sont pas vides
        if cheek_left.size == 0 or cheek_right.size == 0:
            print(f"Régions des joues vides dans la frame {frame_id}")
            return None

        # Redimensionner chaque joue pour garantir un carré
        cheek_left_resized = cv2.resize(cheek_left, (output_size // 2, output_size))
        cheek_right_resized = cv2.resize(cheek_right, (output_size // 2, output_size))

        # Fusionner les joues pour obtenir une image carrée
        combined_cheeks = np.hstack((cheek_left_resized, cheek_right_resized))
        return combined_cheeks

    except Exception as e:
        print(f"Erreur lors de l'extraction des joues à la frame {frame_id}: {e}")
        return None


def normalize_image(image):
    """
    Normalise une image en soustrayant la moyenne et divisant par l'écart-type.
    """
    try:
        image_array = np.array(image).astype(np.float32) / 255.0
        mean = np.mean(image_array, axis=(0, 1))
        std = np.std(image_array, axis=(0, 1)) + 1e-8  # Évite division par zéro
        normalized_image = (image_array - mean) / std
        return (normalized_image * 255).astype(np.uint8)
    except Exception as e:
        print(f"Erreur lors de la normalisation de l'image: {e}")
        return None


def detect_faces(video_path):
    """
    Extrait et normalise les régions des joues de chaque frame d'une vidéo.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Impossible d'ouvrir la vidéo: {video_path}")
            return []

        frames = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Fin de la vidéo à la frame {frame_id}")
                break

            # Conversion en PIL pour le MTCNN
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extraction des joues
            cheeks = extract_cheeks(frame_pil, frame_id, video_path)
            if cheeks is None:
                frame_id += 1
                continue

            # Normalisation
            normalized_cheeks = normalize_image(cheeks)
            if normalized_cheeks is None:
                frame_id += 1
                continue

            frames.append(normalized_cheeks)
            if frame_id % 50 == 0:
                print(f"Frame {frame_id} traitée ({len(frames)} frames valides)")

            frame_id += 1

        cap.release()
        return frames

    except Exception as e:
        print(f"Erreur lors de la détection des visages: {e}")
        return []


def predict_ppg(model, frames, device="cuda"):
    """
    Prédit le signal PPG à partir des frames de visage en utilisant le modèle.
    """
    try:
        model.to(device).eval()

        if len(frames) == 0:
            print("Aucune frame valide pour la prédiction")
            return np.array([])

        # Préparation de la transformation
        transform = transforms.Compose([transforms.ToTensor()])

        # Conversion des frames en tenseurs
        frame_tensors = []
        for frame in frames:
            try:
                # Vérifier que c'est un tableau numpy valide
                if isinstance(frame, np.ndarray) and frame.size > 0:
                    tensor = transform(frame)
                    frame_tensors.append(tensor)
            except Exception as e:
                print(f"Erreur lors de la conversion en tenseur: {e}")
                continue

        if len(frame_tensors) == 0:
            print("Aucune frame n'a pu être convertie en tenseur")
            return np.array([])

        # Empilage des tenseurs
        face_tensors = torch.stack(frame_tensors)

        # Gestion des groupes de 3 frames
        num_frames = face_tensors.shape[0]
        remainder = num_frames % 60

        # Ajout de padding si nécessaire
        if remainder != 0:
            missing = 60 - remainder
            padding = face_tensors[-1:].repeat(missing, 1, 1, 1)
            face_tensors = torch.cat([face_tensors, padding], dim=0)

        num_groups = face_tensors.shape[0] // 60

        # Restructuration pour le modèle 3D
        face_tensors = face_tensors.view(num_groups, 3, 60, 112, 112)

        # Traitement par lots pour économiser la mémoire
        all_predictions = []

        for i in range(0, num_groups, BATCH_SIZE):
            batch = face_tensors[i : i + BATCH_SIZE].to(device)
            with torch.no_grad():
                predictions, attention_maps = model(batch, seq_len=60)
                all_predictions.append(predictions.cpu())

        if not all_predictions:
            return np.array([])

        # Concaténation et conversion en numpy
        ppg_signal = torch.cat(all_predictions, dim=0).numpy().flatten()

        return ppg_signal

    except Exception as e:
        print(f"Erreur lors de la prédiction PPG: {e}")
        return np.array([])


def estimate_heart_rate(ppg_signal, fs=30):
    """
    Estime la fréquence cardiaque à partir du signal PPG.
    """
    try:
        if len(ppg_signal) < fs * 2:  # Au moins 2 secondes de données
            print("Signal PPG trop court pour estimer la fréquence cardiaque")
            return None

        # Filtrage du signal pour isoler les fréquences cardiaques
        filtered_signal = hp.filter_signal(
            ppg_signal,
            [0.7, 3.5],  # Bande passante pour les fréquences cardiaques (42-210 BPM)
            sample_rate=fs,
            order=3,
            filtertype="bandpass",
        )

        # Analyse du signal avec HeartPy
        wd, m = hp.process(filtered_signal, fs)

        # Extraction du BPM moyen
        bpm = m["bpm"]

        return bpm

    except Exception as e:
        print(f"Erreur lors de l'estimation de la fréquence cardiaque: {e}")
        return None


def process_video(video_path, model, device="cuda"):
    """
    Traite une vidéo complète pour estimer la fréquence cardiaque.
    """
    try:
        print(f"Traitement de la vidéo: {video_path}")

        # Extraction des frames de visage
        frames = detect_faces(video_path)

        if len(frames) == 0:
            print(f"Aucune frame valide extraite de {video_path}")
            return video_path, [], None

        print(f"Nombre total de frames valides extraites: {len(frames)}")

        # Prédiction du signal PPG
        ppg_signal = predict_ppg(model, frames, device)

        if len(ppg_signal) == 0:
            print(f"Échec de la génération du signal PPG pour {video_path}")
            return video_path, [], None

        # Estimation de la fréquence cardiaque
        bpm = estimate_heart_rate(ppg_signal)

        # Retourner les résultats
        video_name = os.path.basename(video_path)
        return video_name, ppg_signal, bpm

    except Exception as e:
        print(f"Erreur lors du traitement de la vidéo {video_path}: {e}")
        return os.path.basename(video_path), [], None


def main(video_folder, model_path, output_csv):
    """
    Fonction principale pour traiter toutes les vidéos d'un dossier.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du dispositif: {device}")

    # Chargement du modèle
    model = r3d_transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Liste des vidéos à traiter
    videos = [
        f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov"))
    ]
    print(f"Nombre de vidéos à traiter: {len(videos)}")

    results = {}
    max_length = 0

    # Traitement de chaque vidéo
    for video in videos:
        video_path = os.path.join(video_folder, video)

        try:
            video_name, ppg_signal, bpm = process_video(video_path, model, device)

            if len(ppg_signal) > 0:
                print(f"Vidéo {video_name} traitée avec succès. BPM: {bpm}")
                # Ajouter le BPM à la fin du signal PPG
                results[video_name] = (
                    list(ppg_signal) + [bpm]
                    if bpm is not None
                    else list(ppg_signal) + [None]
                )
                max_length = max(max_length, len(results[video_name]))
            else:
                print(f"Vidéo {video_name} ignorée: aucun signal PPG valide")
        except Exception as e:
            print(f"Erreur lors du traitement de {video}: {e}")

    # Égaliser la longueur des résultats
    for name in results:
        while len(results[name]) < max_length:
            results[name].append(None)

    # Création du DataFrame
    df = pd.DataFrame.from_dict(results, orient="columns")
    df.to_csv(output_csv, index=False)
    print(f"Résultats sauvegardés dans {output_csv}")


if __name__ == "__main__":
    # Configuration des chemins
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_folder = os.path.join(
        project_root, "data/test_dataset"
    )  # Dossier contenant les vidéos
    model_path = os.path.join(
        project_root, "experiments/0020/best_model.pth"
    )  # Chemin du modèle pré-entraîné
    output_csv = os.path.join(project_root, "results.csv")  # Fichier de sortie CSV

    # Exécution du traitement
    main(video_folder, model_path, output_csv)
