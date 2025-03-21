import shutil
import os

# Définition des dossiers
base_dir = os.path.dirname(os.path.abspath(__file__))
source_root = os.path.join(base_dir, "../refined_data")
destination_root = os.path.join(base_dir, "../data_mtcnn")

# Vérifier si le dossier de destination existe, sinon le créer
os.makedirs(destination_root, exist_ok=True)

# Parcourir tous les sous-dossiers dans refined_data/
for video_folder in os.listdir(source_root):
    source_file = os.path.join(source_root, video_folder, "physiological_record.csv")
    destination_file = os.path.join(
        destination_root, video_folder, "physiological_record.csv"
    )  # Renommé pour éviter d'écraser

    # Vérifier si le fichier data.csv existe dans la vidéo
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
        print(f"Copié : {source_file} → {destination_file}")
    else:
        print(f"Fichier non trouvé : {source_file}")
