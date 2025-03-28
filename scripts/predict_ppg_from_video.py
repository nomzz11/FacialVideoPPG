import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lib.pipeline_video.process_video import process_video


def main(video_folder, model_path, output_csv):
    videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    results = {}

    for video in videos:
        video_path = os.path.join(video_folder, video)
        name, ppg_signal, avg_hr = process_video(video_path, model_path)

        if name:
            results[name] = list(ppg_signal) + [avg_hr]

    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv(output_csv, index_label="Video")
    print(f"Résultats sauvegardés dans {output_csv}")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_folder = os.path.join(project_root, "data")
    model_path = os.path.join(project_root, "experiments/0011/best_model.pth")
    main(video_folder, model_path, "predictions.csv")
