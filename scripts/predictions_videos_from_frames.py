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


# Définition du modèle
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
        return x * self.sigmoid(avg_out + max_out)


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
        return x * attn_sigmoid


class CBAM3D(nn.Module):
    def __init__(self, in_planes):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class r3d_transformer_without_attention_map(nn.Module):
    def __init__(self):
        super(r3d_transformer_without_attention_map, self).__init__()
        # Load pre-trained ResNet3D-18 model on videos to extract features
        self.backbone = r3d_18(pretrained=True)

        self.backbone.layer1 = nn.Sequential(self.backbone.layer1, CBAM3D(64))
        self.backbone.layer2 = nn.Sequential(self.backbone.layer2, CBAM3D(128))
        self.backbone.fc = nn.Identity()

        # FC layer to predict final PPG value per frame
        self.fc = nn.Linear(512, 3)

    def forward(self, x):  # x [batch, 3, 3, 112, 112] -> [batch, C, seq_len, H, W]
        x = self.backbone(x)  # x [batch, 512]
        x = self.fc(x)  # x [batch, 3]
        return x


class ChannelAttention_maps(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_maps, self).__init__()
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
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention_maps(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_maps, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_max_out = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(avg_max_out)
        attn_sigmoid = self.sigmoid(attn)
        return x * attn_sigmoid


class CBAM3D_maps(nn.Module):
    def __init__(self, in_planes):
        super(CBAM3D_maps, self).__init__()
        self.ca = ChannelAttention_maps(in_planes)
        self.sa = SpatialAttention_maps()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        attention_map = torch.mean(x, dim=1, keepdim=True)
        return x, attention_map


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
        self.cbam1 = CBAM3D_maps(64)  # 64 canaux après layer1
        self.cbam2 = CBAM3D_maps(128)  # 128 canaux après layer2

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
                outputs = model(batch, seq_len=60)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    predictions, attention_maps = outputs
                else:
                    print(len(outputs))
                    predictions = outputs
                    attention_maps = None
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


def process_video_from_frames(frames_dir, model, device="cuda"):
    """
    Traite les frames d'un dossier pour estimer la fréquence cardiaque.

    Args:
        frames_dir (str): Chemin vers le dossier contenant les frames
        model: Modèle de prédiction
        device (str): Device pour l'inférence ("cuda" ou "cpu")

    Returns:
        tuple: (nom_dossier, signal_ppg, bpm)
    """
    try:
        print(f"Traitement des frames du dossier: {frames_dir}")

        # Chargement des frames depuis le dossier
        frames = []
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

        if not frame_files:
            print(f"Aucune image trouvée dans {frames_dir}")
            return os.path.basename(frames_dir), [], None

        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # Charger l'image directement (pas besoin de detect_faces si les frames sont déjà traitées)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

        if len(frames) == 0:
            print(f"Aucune frame valide chargée depuis {frames_dir}")
            return os.path.basename(frames_dir), [], None

        print(f"Nombre total de frames chargées: {len(frames)}")

        # Si vos frames ne sont pas déjà des visages détectés,
        # vous pouvez appliquer la détection de visage ici
        # face_frames = []
        # for frame in frames:
        #     face = detect_face_in_frame(frame)  # Vous devrez implémenter cette fonction
        #     if face is not None:
        #         face_frames.append(face)
        # frames = face_frames

        # Prédiction du signal PPG
        ppg_signal = predict_ppg(model, frames, device)

        if len(ppg_signal) == 0:
            print(f"Échec de la génération du signal PPG pour {frames_dir}")
            return os.path.basename(frames_dir), [], None

        # Estimation de la fréquence cardiaque
        bpm = estimate_heart_rate(ppg_signal)

        # Retourner les résultats
        dir_name = os.path.basename(frames_dir)
        return dir_name, ppg_signal, bpm

    except Exception as e:
        print(f"Erreur lors du traitement des frames du dossier {frames_dir}: {e}")
        import traceback

        traceback.print_exc()  # Afficher la stack trace pour le débogage
        return os.path.basename(frames_dir), [], None


def main(frames_root_folder, model_path, output_csv):
    """
    Fonction principale pour traiter tous les dossiers de frames.

    Args:
        frames_root_folder (str): Dossier racine contenant les sous-dossiers de frames
        model_path (str): Chemin vers le modèle pré-entraîné
        output_csv (str): Chemin pour sauvegarder les résultats CSV
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du dispositif: {device}")

    # Chargement du modèle
    model = r3d_transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Liste des dossiers de frames à traiter
    frame_folders = [
        d
        for d in os.listdir(frames_root_folder)
        if os.path.isdir(os.path.join(frames_root_folder, d))
    ]
    print(f"Nombre de séquences de frames à traiter: {len(frame_folders)}")

    results = {}
    max_length = 0

    # Traitement de chaque dossier de frames
    for folder in frame_folders:
        folder_path = os.path.join(frames_root_folder, folder)

        try:
            folder_name, ppg_signal, bpm = process_video_from_frames(
                folder_path, model, device
            )

            if len(ppg_signal) > 0:
                print(f"Séquence {folder_name} traitée avec succès. BPM: {bpm}")
                # Ajouter le BPM à la fin du signal PPG
                results[folder_name] = (
                    list(ppg_signal) + [bpm]
                    if bpm is not None
                    else list(ppg_signal) + [None]
                )
                max_length = max(max_length, len(results[folder_name]))
            else:
                print(f"Séquence {folder_name} ignorée: aucun signal PPG valide")
        except Exception as e:
            print(f"Erreur lors du traitement du dossier {folder}: {e}")
            import traceback

            traceback.print_exc()  # Pour obtenir plus de détails sur l'erreur

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
    frames_root_folder = os.path.join(
        project_root, "data_mtcnn"
    )  # Dossier contenant les sous-dossiers de frames
    model_path = os.path.join(
        project_root, "experiments/0020/best_model.pth"
    )  # Chemin du modèle pré-entraîné
    output_csv = os.path.join(project_root, "results.csv")  # Fichier de sortie CSV

    # Exécution du traitement
    main(frames_root_folder, model_path, output_csv)
