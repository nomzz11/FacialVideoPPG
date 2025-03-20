import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.lib.training_cnn3d_transformer import r3d_transformer


def predict_ppg(model, faces, device="cuda"):
    model.to(device).eval()

    transform = transforms.Compose(
        [transforms.Resize((1121, 112)), transforms.ToTensor()]
    )

    face_tensors = torch.stack(
        [transform(Image.fromarray(face)) for face in faces]
    )  # (N, 3, 224, 224)

    num_frames = face_tensors.shape[0]
    num_groups = num_frames // 3

    if num_frames % 3 != 0:
        print(
            f"Attention : {num_frames} frames détectées, tronquées à {num_groups * 3}"
        )
        face_tensors = face_tensors[: num_groups * 3]

    face_tensors = face_tensors.view(num_groups, 3, 3, 224, 224).to(device)

    with torch.no_grad():
        ppg_signal = model(face_tensors).cpu().numpy().flatten()

    return ppg_signal
