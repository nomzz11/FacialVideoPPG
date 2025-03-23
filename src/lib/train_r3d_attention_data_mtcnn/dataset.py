import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class FacialVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        split_strategy="video_length",
        transform=None,
        sequence_length=64,
        seed=42,
    ):
        if split_strategy not in ["video_length", "video_count"]:
            raise ValueError("split_strategy must be 'video_length' ou 'video_count'.")

        self.data_dir = data_dir
        self.split = split
        self.split_strategy = split_strategy
        self.transform = transform
        self.sequence_length = sequence_length
        self.seed = seed

        # Charger les métadonnées
        self.metadata = self._load_metadata()

        # Appliquer le split choisi
        if split_strategy == "video_length":
            self.data = self._split_by_video_length()
        else:
            self.data = self._split_by_video_count()

        # Filtrer les indices valides (éviter les séquences incomplètes)
        self.valid_indices = self._filter_valid_indices()

    def _load_metadata(self):
        """Charge les métadonnées depuis les CSV des vidéos"""
        metadata = []
        for video_folder in os.listdir(self.data_dir):
            csv_path = os.path.join(
                self.data_dir, video_folder, "physiological_record.csv"
            )
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["video_name"] = video_folder  # Associer la vidéo aux frames
                metadata.append(df)
        return pd.concat(metadata, ignore_index=True)

    def _split_by_video_length(self):
        """Divise les vidéos en splits"""
        train_data, val_data, test_data = [], [], []

        for _, group in self.metadata.groupby("video_name"):
            frames = group.sort_values("frame_name")  # Assurer l'ordre temporel
            num_frames = len(frames)

            train_end = int(0.7 * num_frames)
            val_end = train_end + int(0.15 * num_frames)

            train_frames = frames.iloc[:train_end]
            val_frames = frames.iloc[train_end:val_end]
            test_frames = frames.iloc[val_end:]

            train_data.append(train_frames)
            val_data.append(val_frames)
            test_data.append(test_frames)

        df = pd.concat(
            train_data
            if self.split == "train"
            else val_data if self.split == "val" else test_data
        )
        return df

    def _split_by_video_count(self):
        """Divise les vidéos en splits"""
        video_folders = self.metadata["video_name"].unique()

        train_videos, temp_videos = train_test_split(
            video_folders, test_size=0.3, random_state=self.seed
        )
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=0.5, random_state=self.seed
        )

        num_train = len(train_videos)
        num_val = len(val_videos)
        num_test = len(test_videos)

        print(
            f"Splits enregistrés : {num_train} train, {num_val} val, {num_test} test."
        )

        # Vérifier les NaN dans les PPG
        print(
            f"Nombres de NaN dans train: {self.metadata[self.metadata['video_name'].isin(train_videos)]['ppg_value'].isna().sum()}"
        )
        print(
            f"Nombres de NaN dans val: {self.metadata[self.metadata['video_name'].isin(val_videos)]['ppg_value'].isna().sum()}"
        )
        print(
            f"Nombres de NaN dans test: {self.metadata[self.metadata['video_name'].isin(test_videos)]['ppg_value'].isna().sum()}"
        )

        split_videos = (
            train_videos
            if self.split == "train"
            else val_videos if self.split == "val" else test_videos
        )

        return self.metadata[self.metadata["video_name"].isin(split_videos)]

    def _filter_valid_indices(self):
        """Garde uniquement les indices où une séquence complète de `sequence_length` frames est disponible."""
        valid_indices = []
        for i in range(len(self.data) - self.sequence_length + 1):
            # Vérifie que toutes les frames appartiennent à la même vidéo
            video_names = self.data.iloc[i : i + self.sequence_length][
                "video_name"
            ].unique()
            if len(video_names) == 1:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]  # Utilisation des indices filtrés
        sequence = self.data.iloc[start_idx : start_idx + self.sequence_length]

        video_folder = sequence.iloc[0]["video_name"]
        frames = []
        ppg_values = []

        for _, row in sequence.iterrows():
            frame_path = os.path.join(
                self.data_dir, video_folder, f"{row['frame_name']:04d}.jpg"
            )

            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                ppg_values.append(row["ppg_value"])

                if pd.isna(row["ppg_value"]):
                    print(
                        f"NaN détecté pour {row['video_name']} à la frame {row['frame_name']}"
                    )
        return torch.stack(frames), torch.tensor(ppg_values)
