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
        split_strategy=None,
        transform=None,
        sequence_length=3,
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
            print(f"[{self.split}] Nombre total de frames après split:", len(self.data))
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
        print(self.metadata["video_name"].value_counts())
        video_folders = self.metadata["video_name"].unique()
        print(len(video_folders))
        train_videos, temp_videos = train_test_split(
            video_folders, test_size=0.3, random_state=self.seed
        )
        val_videos, test_videos = train_test_split(
            temp_videos, test_size=0.5, random_state=self.seed
        )

        if self.split == "train":
            return self.metadata[
                self.metadata["video_name"].isin(train_videos).reset_index(drop=True)
            ]
        elif self.split == "val":
            return self.metadata[
                self.metadata["video_name"].isin(val_videos).reset_index(drop=True)
            ]
        elif self.split == "test":
            return self.metadata[
                self.metadata["video_name"].isin(test_videos).reset_index(drop=True)
            ]

    def _generate_samples(self):
        sequences = []
        for video_name, group in self.grouped_data:
            frames = group.sort_values("frame_name")
            frame_indices = list(frames.index)
            for i in range(len(frame_indices) - self.sequence_length + 1):
                sequences.append(frame_indices[i : i + self.sequence_length])
        return sequences

    def detect_face(self, image, video_name):
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

            """
            if video_name in self.video_stats:
                mean, std = self.video_stats[video_name]
                face_crop = face_crop.astype(np.float32)
                face_crop = (face_crop - mean) / (std + 1e-8)  # Évite div/0
                face_crop = np.clip(face_crop * 255, 0, 255).astype(np.uint8)"
            """

            return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        return image

    def extract_cheeks(frame, output_size=112):
        """
        Détecte précisément les joues gauche et droite depuis une frame avec MTCNN.
        Retourne une image carrée combinant les deux joues.
        """
        mtcnn = MTCNN(
            select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

        if landmarks is None or len(landmarks) == 0:
            print("Aucun visage détecté")
            return None

        landmarks = landmarks[0]

        # Landmarks précis pour joues
        left_eye, right_eye, _, mouth_left, mouth_right = landmarks

        h, w, _ = frame.shape

        # Joues Gauche
        cheek_left_x1 = int(
            max(mouth_left[0] - 0.6 * abs(mouth_left[0] - left_eye[0]), 0)
        )
        cheek_left_x2 = int(mouth_left[0])
        cheek_left_y1 = int(left_eye[1])
        cheek_left_y2 = int(mouth_left[1] + 0.2 * abs(mouth_left[1] - left_eye[1]))
        cheek_left_y2 = min(cheek_left_y2, h)

        left_cheek = frame[cheek_left_y1:cheek_left_y2, cheek_left_x1:cheek_left_x2]

        # Joue Droite
        cheek_right_x1 = int(mouth_right[0])
        cheek_right_x2 = int(
            min(mouth_right[0] + 0.6 * abs(right_eye[0] - mouth_right[0]), w)
        )
        cheek_right_y1 = int(right_eye[1])
        cheek_right_y2 = int(mouth_right[1] + 0.2 * abs(mouth_right[1] - right_eye[1]))
        cheek_right_y2 = min(cheek_right_y2, h)

        right_cheek = frame[
            cheek_right_y1:cheek_right_y2, cheek_right_x1:cheek_right_x2
        ]

        # Vérification de la taille des joues
        if left_cheek.size == 0 or right_cheek.size == 0:
            print("Erreur extraction joues, vérifiez la détection des landmarks.")
            return None

        # Redimensionnement des joues
        left_cheek_resized = cv2.resize(left_cheek, (output_size, output_size))
        right_cheek_resized = cv2.resize(right_cheek, (output_size, output_size))

        # Combinaison claire des joues côte-à-côte
        combined_cheeks = np.hstack((left_cheek_resized, right_cheek_resized))

        # Redimension finale pour un carré parfait
        combined_cheeks_square = cv2.resize(combined_cheeks, (output_size, output_size))

        return combined_cheeks_square

    def _normalize_image(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1)) + 1e-8  # Évite division par zéro
        normalized_image = (image - mean) / std
        return Image.fromarray((normalized_image * 255).astype(np.uint8))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        frame_indices = self.samples[id]  # Liste de `sequence_length` indices
        frames = []
        ppg_values = []

        for frame_idx in frame_indices:
            row = self.data.iloc[frame_idx]
            video_folder = row["video_name"]
            frame_name = f"{row['frame_name']:04d}.jpg"
            frame_path = os.path.join(self.data_dir, video_folder, frame_name)
            ppg_value = row["ppg_value"]

            frame = Image.open(frame_path).convert("RGB")

            # frame = cv2.imread(frame_path)
            composite_image = self.extract_cheeks(frame)

            if composite_image is None:
                continue

            normalized_image = self._normalize_image(composite_image)

            if self.transform:
                normalized_image = self.transform(normalized_image)

            frames.append(normalized_image)
            ppg_values.append(ppg_value)

        frames = torch.stack(frames)  # Shape: (seq_len, C, H, W) -> 4D tensor
        frames = frames.permute(1, 0, 2, 3)  # Shape: (C, seq_len, H, W)
        ppg_values = torch.tensor(ppg_values, dtype=torch.float32)  # Shape: (seq_len,)

        return frames, ppg_values
