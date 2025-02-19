import os, sys
import pandas as pd
import numpy as np
import json
import torch.optim as optim
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import get_cli_options_training
from src.lib.training import (
    FacialVideoDataset,
    split_dataset,
    dataloader,
    ResNetPPG,
    train_model,
)


def get_next_job_id(save_dir="experiments"):
    os.makedirs(save_dir, exist_ok=True)
    existing_jobs = sorted([d for d in os.listdir(save_dir) if d.isdigit()])
    return f"{int(existing_jobs[-1]) + 1:04d}" if existing_jobs else "0000"


if __name__ == "__main__":

    cli_options = get_cli_options_training()

    job_id = get_next_job_id()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_path = os.path.join(project_root, "experiments", job_id)
    os.makedirs(save_path, exist_ok=True)

    dataset = FacialVideoDataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, split_strategy=cli_options["split_strategy"]
    )
    train_dataloader, val_dataloader, test_dataloader = dataloader(
        train_dataset, val_dataset, test_dataset
    )

    model = ResNetPPG()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cli_options["lr"],
        weight_decay=cli_options["weight_decay"],
    )

    config = {
        "job_id": job_id,
        "model": "ResNetPPG",
        "epochs": cli_options["epochs"],
        "learning_rate": cli_options["lr"],
        "weight_decay": cli_options["weight_decay"],
        "loss_function": "MSELoss",
        "optimizer": "Adam",
        "dataset_split_strategy": cli_options["split_strategy"],
        "batch_size": train_dataloader.batch_size,
    }

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Configuration sauvegardée dans {config_path}")

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        save_path,
        epochs=cli_options["epochs"],
    )
