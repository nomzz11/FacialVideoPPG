import torch
import numpy as np
from torchvision import transforms


def predict_ppg(model, faces, device="cuda"):
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Redimensionne toutes les images à 224x224
            transforms.ToTensor(),  # Convertit en tenseur
        ]
    )

    # Transforme toutes les images en tenseurs
    face_tensors = torch.stack([transform(face) for face in faces])  # (N, 3, 224, 224)

    # Vérifie que le nombre de frames est un multiple de 3
    num_frames = face_tensors.shape[0]
    num_groups = num_frames // 3  # Nombre de groupes de 3 frames

    if num_frames % 3 != 0:
        print(
            f"Attention : {num_frames} frames détectées, tronquées à {num_groups * 3}"
        )
        face_tensors = face_tensors[: num_groups * 3]  # Tronquer à un multiple de 3

    # Regroupe en séquences de 3 frames
    face_tensors = face_tensors.view(
        num_groups, 3, 3, 224, 224
    )  # (batch, 3, 3, 224, 224)

    # Envoi sur le device
    face_tensors = face_tensors.to(device)

    # Prédiction sans calcul de gradients
    with torch.no_grad():
        ppg_preds = model(face_tensors).cpu().numpy().flatten()

    return ppg_preds
