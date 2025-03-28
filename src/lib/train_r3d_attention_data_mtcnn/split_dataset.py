import os
from torchvision import transforms
from .dataset import FacialVideoDataset


def split_dataset(dataset, split_strategy, seq_len):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../../../data_mtcnn")

    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = dataset(
        data_dir,
        seq_len,
        split="train",
        split_strategy=split_strategy,
        transform=transform,
    )
    val_dataset = dataset(
        data_dir,
        seq_len,
        split="val",
        split_strategy=split_strategy,
        transform=transform,
    )
    test_dataset = dataset(
        data_dir,
        seq_len,
        split="test",
        split_strategy=split_strategy,
        transform=transform,
    )

    return train_dataset, val_dataset, test_dataset
