import os, sys
import pandas as pd
import numpy as np
import json
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import get_cli_options_training
from src.lib.train_r3d_attention_data_mtcnn import train_model
from src.lib.train_r3d_attention_data_mtcnn import (
    FacialVideoDataset,
    split_dataset,
    dataloader,
    CCCLoss,
)
from src.lib.training_cnn3d_transformer import (
    r3d_transformer,
    PearsonLoss,
)


def get_next_job_id(save_dir="experiments"):
    os.makedirs(save_dir, exist_ok=True)
    existing_jobs = sorted([d for d in os.listdir(save_dir) if d.isdigit()])
    if existing_jobs:
        last_job_id = existing_jobs[-1]
        last_job_path = os.path.join(save_dir, last_job_id)
        if not os.listdir(last_job_path):
            return last_job_id

        return f"{int(existing_jobs[-1]) + 1:04d}"
    return "0000"


if __name__ == "__main__":

    cli_options = get_cli_options_training()

    model = r3d_transformer(out_features=cli_options["seq_len"])

    job_id = get_next_job_id()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(project_root, "experiments", job_id)
    os.makedirs(save_path, exist_ok=True)

    dataset = FacialVideoDataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        split_strategy=cli_options["split_strategy"],
        seq_len=cli_options["seq_len"],
    )
    train_dataloader, val_dataloader, test_dataloader = dataloader(
        train_dataset, val_dataset, test_dataset, cli_options["batch_size"]
    )

    loss = cli_options["loss"]
    mse_loss = nn.MSELoss()
    pearson_loss = PearsonLoss()
    ccc_loss = CCCLoss()

    alpha = 0.8  # Poids pour la MSE
    beta = 0.2  # Poids pour la Pearson Loss

    def combined_MSE_Pearson_loss(pred, target):
        loss_mse = mse_loss(pred, target)
        loss_pearson = pearson_loss(pred, target)
        return alpha * loss_mse + beta * loss_pearson

    def combined_MSE_CCC_loss(pred, target):
        loss_mse = mse_loss(pred, target)
        loss_ccc = ccc_loss(pred, target)
        return alpha * loss_mse + beta * loss_ccc

    if loss == "mse":
        criterion = mse_loss
    elif loss == "mse_ccc":
        criterion = combined_MSE_CCC_loss
    elif loss == "pearson":
        criterion = pearson_loss
    else:
        raise ValueError("loss must be 'mse' , 'mse_ccc' or 'mse_pearson'.")

    optimizer = optim.Adam(
        model.parameters(),
        lr=cli_options["lr"],
        weight_decay=cli_options["weight_decay"],
    )

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        save_path,
        seq_len=cli_options["seq_len"],
        epochs=cli_options["epochs"],
        gpu=cli_options["gpu"],
    )

    config = {
        "job_id": job_id,
        "model": cli_options["model"],
        "epochs": cli_options["epochs"],
        "learning_rate": cli_options["lr"],
        "weight_decay": cli_options["weight_decay"],
        "loss_function": cli_options["loss"],
        "optimizer": "Adam",
        "dataset_split_strategy": cli_options["split_strategy"],
        "seq_len": cli_options["seq_len"],
        "batch_size": train_dataloader.batch_size,
    }

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration sauvegardée dans {config_path}")
