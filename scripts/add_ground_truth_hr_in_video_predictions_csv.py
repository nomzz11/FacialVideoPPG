import pandas as pd
import json
import os
import numpy as np
from glob import glob


def add_hr_targets_to_predictions(
    predictions_csv_path, json_directory, output_csv_path=None
):
    """
    Ajoute les valeurs cibles HR depuis les fichiers JSON aux prédictions CSV.
    Utilise une correspondance directe entre les noms de vidéos du CSV et
    les noms de fichiers extraits du JSON.
    """
    # Charger le CSV des prédictions
    print(f"Chargement du CSV de prédictions: {predictions_csv_path}")
    df_predictions = pd.read_csv(predictions_csv_path)

    # Si aucun chemin de sortie n'est spécifié, écraser le fichier original
    if output_csv_path is None:
        output_csv_path = predictions_csv_path

    # Obtenir la liste unique des noms de vidéos
    video_names = df_predictions.iloc[0].tolist()
    unique_videos = list(set(video_names))
    print(f"Nombre de vidéos uniques dans le CSV: {len(unique_videos)}")

    # Dictionnaire pour stocker les moyennes HR par vidéo
    video_hr_means = {}

    # Charger tous les fichiers JSON et extraire les HR moyens
    json_files = glob(os.path.join(json_directory, "*.json"))
    print(f"Nombre de fichiers JSON trouvés: {len(json_files)}")

    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                physiological_records = json.load(f)

            # Traiter chaque scénario
            for scenario in physiological_records["scenarios"]:
                # Vérifier les conditions du scénario comme dans votre code original
                scenario_settings = scenario["scenario_settings"]
                if (
                    scenario_settings["position"] != "Sitting"
                    or scenario_settings["facial_movement"] != "No movement"
                    or scenario_settings["talking"] != "N"
                ):
                    continue

                # Extraire le nom de la vidéo exactement comme dans votre code
                if (
                    "RGB" in scenario["recordings"]
                    and "filename" in scenario["recordings"]["RGB"]
                ):
                    video_name = scenario["recordings"]["RGB"]["filename"].split(".")[0]
                else:
                    # Passer au scénario suivant si pas de nom de vidéo
                    continue

                # Vérifier si cette vidéo est dans notre CSV
                if video_name not in unique_videos:
                    continue

                # Extraire les données HR pour ce scénario
                if "hr" in scenario["recordings"]:
                    hr_data = scenario["recordings"]["hr"]
                    if "timeseries" in hr_data:
                        hr_values = [ts[1] for ts in hr_data["timeseries"]]

                        if hr_values:
                            # Calculer la moyenne des HR pour ce scénario
                            hr_mean = np.mean(hr_values)
                            video_hr_means[video_name] = hr_mean
                            print(f"Vidéo {video_name}: HR moyen = {hr_mean:.2f}")

        except Exception as e:
            print(f"Erreur lors du traitement du fichier {json_path}: {e}")

    # Créer une nouvelle ligne pour les valeurs HR cibles
    hr_targets = []
    for video_name in video_names:
        hr_targets.append(video_hr_means.get(video_name, np.nan))

    # Ajouter la ligne des HR cibles au DataFrame
    df_predictions.loc[len(df_predictions)] = hr_targets

    # Vérifier combien de vidéos ont reçu une valeur HR cible
    videos_with_hr = sum(1 for v in unique_videos if v in video_hr_means)
    print(f"Vidéos associées à une HR cible: {videos_with_hr}/{len(unique_videos)}")

    # Vérifier s'il y a des vidéos dans le CSV qui n'ont pas trouvé de correspondance
    videos_without_hr = [v for v in unique_videos if v not in video_hr_means]
    if videos_without_hr:
        print(
            f"Vidéos sans HR cible ({len(videos_without_hr)}): {videos_without_hr[:5]}..."
        )
        if len(videos_without_hr) > 5:
            print("(et d'autres...)")

    # Sauvegarder le CSV enrichi
    df_predictions.to_csv(output_csv_path, index=False)
    print(f"CSV enrichi sauvegardé à: {output_csv_path}")

    return df_predictions


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
predictions_csv_path = os.path.join(
    project_root, "experiments/0020/video_predictions.csv"
)
json_directory = os.path.join(project_root, "data")
output_csv_path = os.path.join(
    project_root, "experiments/0020/video_predictions_with_hr.csv"
)

df_enriched = add_hr_targets_to_predictions(
    predictions_csv_path, json_directory, output_csv_path
)
