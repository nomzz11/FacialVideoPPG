import shutil
import os
import pandas as pd

# Définition des dossiers
base_dir = os.path.dirname(os.path.abspath(__file__))
source_root = os.path.join(base_dir, "../refined_data")
destination_root = os.path.join(base_dir, "../data_mtcnn")

# Vérifier si le dossier de destination existe, sinon le créer
os.makedirs(destination_root, exist_ok=True)

# Parcourir tous les sous-dossiers dans refined_data/
for video_folder in os.listdir(source_root):
    source_file = os.path.join(source_root, video_folder, "physiological_record.csv")
    destination_folder = os.path.join(destination_root, video_folder)
    destination_file = os.path.join(destination_folder, "physiological_record.csv")

    # Vérifier si le fichier CSV source existe
    if os.path.exists(source_file):
        # Créer le dossier de destination s'il n'existe pas
        os.makedirs(destination_folder, exist_ok=True)

        # Lire le CSV source
        df = pd.read_csv(source_file)

        # Liste pour stocker les indices des lignes à conserver
        rows_to_keep = []
        # Dictionnaire pour stocker les noms de frames formatés
        formatted_frames = {}

        # Vérifier pour chaque ligne si la frame existe
        for index, row in df.iterrows():
            # Supposons que le nom de la colonne contenant le nom de frame est 'frame_name'
            if "frame_name" in df.columns:
                # Formater le nom de la frame avec 4 chiffres
                try:
                    # Si frame_name est un nombre, le formater en 4 chiffres
                    frame_num = int(row["frame_name"])
                    formatted_frame_name = f"{frame_num:04d}"
                    frame_file = formatted_frame_name + ".jpg"
                except (ValueError, TypeError):
                    # Si ce n'est pas un nombre, utiliser comme tel
                    formatted_frame_name = str(row["frame_name"])
                    frame_file = formatted_frame_name + ".jpg"

                frame_path = os.path.join(destination_folder, frame_file)

                # Si la frame existe, conserver la ligne et stocker le nom formaté
                if os.path.exists(frame_path):
                    rows_to_keep.append(index)
                    formatted_frames[index] = formatted_frame_name
            else:
                # Si la colonne 'frame_name' n'existe pas, on garde toutes les lignes
                rows_to_keep = list(range(len(df)))
                break

        # Filtrer le DataFrame pour ne garder que les lignes dont les frames existent
        filtered_df = df.iloc[rows_to_keep].copy()

        # Mettre à jour les noms de frames dans le DataFrame filtré
        # Avant de mettre à jour les noms de frames dans le DataFrame filtré
        if "frame_name" in filtered_df.columns and len(formatted_frames) > 0:
            # Convertir la colonne en type 'string' ou 'object'
            filtered_df["frame_name"] = filtered_df["frame_name"].astype(str)

            # Maintenant mettre à jour les valeurs
            for idx, formatted_name in formatted_frames.items():
                filtered_df.loc[filtered_df.index == idx, "frame_name"] = formatted_name

        # Enregistrer le CSV filtré
        filtered_df.to_csv(destination_file, index=False)

        print(
            f"CSV filtré créé : {destination_file} ({len(filtered_df)}/{len(df)} lignes)"
        )
    else:
        print(f"Fichier CSV source non trouvé : {source_file}")
