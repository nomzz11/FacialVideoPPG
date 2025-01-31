import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lib.training import FacialVideoDataset, split_dataset, dataloader


if __name__ == "__main__":

    dataset = FacialVideoDataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
