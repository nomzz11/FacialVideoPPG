import os, pandas as pd, numpy as np, cv2, torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
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
        if split_strategy not in ["video_length", "video_count"]:
            raise ValueError("split_strategy must be 'video_length' ou 'video_count'.")

        self.data_dir = data_dir
        self.split = split
        self.split_strategy = split_strategy
        self.transform = transform
        self.seed = seed

        # Charger les métadonnées
        self.metadata = self._load_metadata()

        # Appliquer le split choisi
        if split_strategy == "video_length":
            self.data = self._split_by_video_length()
        else:
            self.data = self._split_by_video_count()

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

        split_videos = (
            train_videos
            if self.split == "train"
            else val_videos if self.split == "val" else test_videos
        )
        return self.metadata[self.metadata["video_name"].isin(split_videos)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from src.lib.training_cnn3d_transformer import extract_cheeks

        """Charge une frame et sa valeur PPG"""
        row = self.data.iloc[idx]
        video_name = row["video_name"]
        frame_name = f"{row['frame_name']:04d}.jpg"
        frame_path = os.path.join(self.data_dir, video_name, frame_name)

        frame = Image.open(frame_path).convert("RGB")
        # composite_image = extract_cheeks(frame, idx, video_name)
        # composite_image = Image.fromarray(composite_image)

        # Normalisation
        composite_image = self._normalize_image(frame)
        if self.transform:
            composite_image = self.transform(composite_image)

        frame_tensor = torch.tensor(
            np.array(composite_image), dtype=torch.float32
        ).permute(
            2, 0, 1
        )  # (C, H, W)
        ppg_value = torch.tensor(row["ppg_value"], dtype=torch.float32)  # (1,)

        if self.video_is_invalid(
            idx
        ):  # Méthode qui vérifie une condition personnalisée pour cette vidéo
            print(f"Vidéo {idx} ignorée.")
            return None

        if frame_tensor is None or ppg_value is None:
            return None

        return frame_tensor, ppg_value, video_name

    def _normalize_image(self, image):
        """Normalise l'image pour éviter les biais d'éclairage"""
        image = np.array(image).astype(np.float32) / 255.0
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1)) + 1e-8  # Évite la division par zéro
        normalized_image = (image - mean) / std
        return Image.fromarray((normalized_image * 255).astype(np.uint8))

    def video_is_invalid(self, idx):
        # Critère pour ignorer une vidéo en particulier
        invalid_videos = [
            "ba17778fd8c441659d2c9d0142153c5d_1",
            "ba17778fd8c441659d2c9d0142153c5d_2",
        ]  # Liste d'IDs ou de noms de vidéos à ignorer
        video_name = self.data.iloc[idx]["video_name"]

        if video_name in invalid_videos:
            return True
        return False
