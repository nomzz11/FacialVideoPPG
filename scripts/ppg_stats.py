import pandas as pd
import os


class PPGStatistics:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_statistics_per_video(self):
        """Calcule les statistiques des valeurs PPG pour chaque vidéo."""
        all_stats = []

        video_folders = os.listdir(self.data_dir)
        for video_folder in video_folders:
            video_path = os.path.join(self.data_dir, video_folder)
            if os.path.isdir(video_path):
                csv_path = os.path.join(video_path, "physiological_record.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if "ppg_value" in df.columns:
                        ppg_values = df["ppg_value"].dropna().tolist()
                        if ppg_values:
                            ppg_series = pd.Series(ppg_values)
                            stats = {
                                "Vidéo": video_folder,
                                "Min": ppg_series.min(),
                                "Max": ppg_series.max(),
                                "Moyenne": ppg_series.mean(),
                                "Médiane": ppg_series.median(),
                                "Écart-type": ppg_series.std(),
                                "1er Quartile (Q1)": ppg_series.quantile(0.25),
                                "3e Quartile (Q3)": ppg_series.quantile(0.75),
                            }
                            all_stats.append(stats)

        return all_stats

    def save_statistics_to_csv(self):
        """Sauvegarde les statistiques PPG par vidéo dans un fichier CSV."""
        stats_list = self.compute_statistics_per_video()
        output_path = os.path.join(self.output_dir, "ppg_statistics_per_video.csv")
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            stats_df.to_csv(output_path, index=False)
            print(f"Statistiques PPG enregistrées dans {output_path}")
        else:
            print("Aucune donnée PPG trouvée.")


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "../refined_data")
output_dir = os.path.join(os.path.dirname(data_dir), "stats")

ppg_stats = PPGStatistics(data_dir, output_dir)
ppg_stats.save_statistics_to_csv()
