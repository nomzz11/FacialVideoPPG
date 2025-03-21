import os
import torch

# Chemin du dossier contenant les vidéos (chaque dossier = 1 vidéo)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_folder = os.path.join(project_root, "data")
data_dir = os.path.join(project_root, "data_mtcnn")

# Liste des dossiers (vidéos) triée
video_folders = sorted(os.listdir(data_dir))

# Vérification
num_videos = len(video_folders)
print(f"Nombre total de vidéos : {num_videos}")

# Définition des proportions
num_train = int(0.7 * num_videos)
num_val = int(0.15 * num_videos)
num_test = num_videos - (num_train + num_val)  # Gère l'arrondi

# Création des splits
train_videos = video_folders[:num_train]
val_videos = video_folders[num_train : num_train + num_val]
test_videos = video_folders[num_train + num_val :]

# Sauvegarde des splits
splits = {"train": train_videos, "val": val_videos, "test": test_videos}
torch.save(splits, "splits.pth")

print(f"Splits enregistrés : {num_train} train, {num_val} val, {num_test} test.")
