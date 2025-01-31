import os, pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


class FacialVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        split_strategy="video_length",
        transform=None,
        seed=42,
    ):
        self.data_dir = data_dir
        self.split = split
        self.split_strategy = split_strategy
        self.transform = transform
        self.seed = seed

        # Load metadata from CSV files
        self.metadata = self._load_metadata()

        # Perform splitting based on the selected strategy
        if split_strategy == "video_length":
            self.data = self._split_by_video_length()
        elif split_strategy == "video_count":
            self.data = self._split_by_video_count()
        else:
            raise ValueError(
                "Invalid split_strategy. Use 'video_length' or 'video_count'."
            )

    def _load_metadata(self):
        metadata = []
        video_folders = os.listdir(self.data_dir)
        for video_folder in video_folders:
            video_path = os.path.join(self.data_dir, video_folder)
            if os.path.isdir(video_path):
                csv_path = os.path.join(video_path, "physiological_record.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df["video_folder"] = video_folder
                    metadata.append(df)
        return pd.concat(metadata, ignore_index=True)

    def _split_by_video_length(self):
        grouped = self.metadata.groupby("video_folder")
        train_data, val_data, test_data = [], [], []

        for _, group in grouped:
            # Sort frames by frame_name to maintain order
            frames = group.sort_values("frame_name")

            # Split frames into train, validation, and test
            train, temp = train_test_split(
                frames, test_size=0.3, random_state=self.seed, shuffle=False
            )
            val, test = train_test_split(
                temp, test_size=0.5, random_state=self.seed, shuffle=False
            )

            if self.split == "train":
                train_data.append(train)
            elif self.split == "val":
                val_data.append(val)
            elif self.split == "test":
                test_data.append(test)

        if self.split == "train":
            return pd.concat(train_data, ignore_index=True)
        elif self.split == "val":
            return pd.concat(val_data, ignore_index=True)
        elif self.split == "test":
            return pd.concat(test_data, ignore_index=True)

    def _split_by_video_count(self):
        video_folders = self.metadata["video_folder"].unique()
        train_videos, temp_videos = train_test_split(
            video_folders, test_size=0.3, random_state=self.seed
        )
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=0.5, random_state=self.seed
        )

        if self.split == "train":
            return self.metadata[self.metadata["video_folder"].isin(train_videos)]
        elif self.split == "val":
            return self.metadata[self.metadata["video_folder"].isin(val_videos)]
        elif self.split == "test":
            return self.metadata[self.metadata["video_folder"].isin(test_videos)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        row = self.data.iloc[id]
        video_folder = row["video_folder"]
        frame_name = f"{row['frame_name']:04d}.jpg"
        frame_path = os.path.join(self.data_dir, video_folder, frame_name)
        ppg_value = row["ppg_value"]

        image = Image.open(frame_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, ppg_value
