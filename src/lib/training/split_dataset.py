import os
from torchvision import transforms
from .dataset import FacialVideoDataset


def split_dataset(dataset):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../../../refined_data")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = dataset(
        data_dir, split="train", split_strategy="video_length", transform=transform
    )
    val_dataset = dataset(
        data_dir, split="val", split_strategy="video_length", transform=transform
    )
    test_dataset = dataset(
        data_dir, split="test", split_strategy="video_length", transform=transform
    )

    return train_dataset, val_dataset, test_dataset
