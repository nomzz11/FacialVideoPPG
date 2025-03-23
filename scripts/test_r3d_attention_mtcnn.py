import os, sys
import pandas as pd
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import get_cli_options_test
from src.lib.test_r3d_attention_mtcnn import test_model, FacialVideoDataset
from src.lib.train_r3d_attention_data_mtcnn import (
    split_dataset,
    dataloader,
)
from src.lib.training_cnn3d_transformer import r3d_transformer, PearsonLoss
from src.lib.pipeline_video.estimate_heart_rate import estimate_heart_rate


if __name__ == "__main__":

    cli_options = get_cli_options_test()

    device = "cuda:0"
    if cli_options["gpu"] == 1:
        device = "cuda:1"

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    job_id = cli_options["model_log"]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(project_root, "experiments", job_id)
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(project_root, "experiments", job_id, "best_model.pth")
    model = r3d_transformer(out_features=cli_options["seq_len"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = FacialVideoDataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        split_strategy="video_count",
        seq_len=cli_options["seq_len"],
    )
    train_dataloader, val_dataloader, test_dataloader = dataloader(
        train_dataset, val_dataset, test_dataset, cli_options["batch_size"]
    )

    criterion = PearsonLoss()

    video_preds = test_model(
        model,
        test_dataloader,
        criterion,
        save_path,
        device,
        seq_len=cli_options["seq_len"],
    )

    bpm_results = {
        video_name: estimate_heart_rate(preds)
        for video_name, preds in video_preds.items()
    }

    df_dict = {}

    for video_name, preds in video_preds.items():
        df_dict[video_name] = (
            preds  # Chaque vidéo devient une colonne avec ses prédictions en ligne
        )

    # Conversion en DataFrame
    df = pd.DataFrame.from_dict(df_dict, orient="index").transpose()

    # Ajout des BPM estimés à partir des prédictions
    bpm_df = pd.DataFrame([bpm_results], index=["BPM"])

    # Concaténation du DataFrame des prédictions et des BPM
    df = pd.concat([df, bpm_df])

    # Définition du chemin de sauvegarde
    csv_path = os.path.join(save_path, "video_predictions.csv")

    # Sauvegarde en CSV
    df.to_csv(csv_path, index=False)

    print(f"Fichier CSV des prédictions sauvegardé à {csv_path}")
