import os, pandas as pd, numpy as np, cv2, torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


class FacialVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        split_strategy=None,
        transform=None,
        sequence_length=30,
        use_lstm=False,
        seed=42,
    ):
        if split_strategy not in ["video_length", "video_count"]:
            raise ValueError(
                "split_strategy must be specified as 'video_length' or 'video_count'."
            )

        self.data_dir = data_dir
        self.split = split
        self.split_strategy = split_strategy
        self.transform = transform
        self.sequence_length = sequence_length
        self.use_lstm = use_lstm
        self.seed = seed

        # Load metadata from CSV files
        self.metadata = self._load_metadata()

        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Perform splitting based on the selected strategy
        if split_strategy == "video_length":
            self.data = self._split_by_video_length()
        elif split_strategy == "video_count":
            self.data = self._split_by_video_count()
        else:
            raise ValueError(
                "Invalid split_strategy. Use 'video_length' or 'video_count'."
            )

        # Grouper les frames par vidéo pour former des séquences
        self.grouped_data = self.data.groupby("video_name")

        # Liste des échantillons possibles (séquences ou frames uniques)
        self.samples = self._generate_samples()

    def _load_metadata(self):
        metadata = []
        video_folders = os.listdir(self.data_dir)
        for video_folder in video_folders:
            video_path = os.path.join(self.data_dir, video_folder)
            if os.path.isdir(video_path):
                csv_path = os.path.join(video_path, "physiological_record.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    metadata.append(df)
        return pd.concat(metadata, ignore_index=True)

    def _split_by_video_length(self):
        grouped = self.metadata.groupby("video_name")
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
        video_folders = self.metadata["video_name"].unique()
        train_videos, temp_videos = train_test_split(
            video_folders, test_size=0.3, random_state=self.seed
        )
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=0.5, random_state=self.seed
        )

        if self.split == "train":
            return self.metadata[self.metadata["video_name"].isin(train_videos)]
        elif self.split == "val":
            return self.metadata[self.metadata["video_name"].isin(val_videos)]
        elif self.split == "test":
            return self.metadata[self.metadata["video_name"].isin(test_videos)]

    def _generate_samples(self):
        """Generate indices based on mode (ResNet alone or LSTM)"""
        if self.use_lstm:
            sequences = []
            for video_name, group in self.grouped_data:
                frames = group.sort_values("frame_name")
                frame_indices = list(frames.index)
                for i in range(len(frame_indices) - self.sequence_length + 1):
                    sequences.append(frame_indices[i : i + self.sequence_length])
            return sequences
        else:
            return self.data.index.tolist()

    def detect_face(self, image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100)
        )

        if len(faces) > 0:
            x, y, w, h = sorted(
                faces, key=lambda rect: rect[2] * rect[3], reverse=True
            )[0]
            face_crop = image_cv[y : y + h, x : x + w]
            return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        if self.use_lstm:
            frame_indices = self.samples[id]  # Liste de `sequence_length` indices
            frames = []
            ppg_values = []

            for frame_idx in frame_indices:
                row = self.data.iloc[frame_idx]
                video_folder = row["video_name"]
                frame_name = f"{row['frame_name']:04d}.jpg"
                frame_path = os.path.join(self.data_dir, video_folder, frame_name)
                ppg_value = row["ppg_value"]

                image = Image.open(frame_path).convert("RGB")
                image = self.detect_face(image)  # Appliquer la détection du visage

                if self.transform:
                    image = self.transform(image)

                frames.append(image)
                ppg_values.append(ppg_value)

            frames = torch.stack(frames)  # Shape: (seq_len, C, H, W)
            ppg_values = torch.tensor(
                ppg_values, dtype=torch.float32
            )  # Shape: (seq_len,)

            return frames, ppg_values

        else:

            row = self.data.iloc[id]
            video_folder = row["video_name"]
            frame_name = f"{row['frame_name']:04d}.jpg"
            frame_path = os.path.join(self.data_dir, video_folder, frame_name)
            ppg_value = row["ppg_value"]

            image = Image.open(frame_path).convert("RGB")

            image = self.detect_face(image)

            if self.transform:
                image = self.transform(image)

            return image, ppg_value
