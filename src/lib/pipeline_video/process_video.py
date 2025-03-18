import torch, matplotlib.pyplot as plt, numpy as np, heartpy as hp
from src.lib.training_cnn3d_transformer import r3d_transformer
from src.lib.pipeline_video.face_detection import face_detection
from src.lib.pipeline_video.predict_ppg import predict_ppg
from src.lib.pipeline_video.estimate_heart_rate import estimate_heart_rate
from src.lib.ppg_processing.filtered_ppg import bandpass_filter


def process_video(video_path, model_path, save_path="plotsHR/heart_rate.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = r3d_transformer()  # Adapter selon ton modèle
    model.load_state_dict(torch.load(model_path, map_location=device))

    faces = face_detection(video_path)
    if faces is None:
        print("Aucun visage détecté.")
        return

    ppg_signal = predict_ppg(model, faces, device)
    print("Min:", np.min(ppg_signal), "Max:", np.max(ppg_signal))
    print(len(ppg_signal))
    # filtered_ppg_signal = bandpass_filter(ppg_signal, fs=30)

    wd, m, heart_rates = estimate_heart_rate(ppg_signal)

    # Graphique du PPG
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ppg_signal, label="PPG brut", alpha=0.6)
    # plt.plot(ppg_signal, label="PPG filtré", alpha=0.8)
    plt.title("Signal PPG extrait")
    plt.legend()
    plt.savefig(save_path)

    if heart_rates is None:
        print("L'estimation de la fréquence cardiaque a échoué.")
        return
    plt.subplot(2, 1, 2)
    plt.plot(
        heart_rates,
        marker="o",
        linestyle="-",
        color="red",
        label="Heart Rate (BPM)",
    )
    plt.xlabel("Temps (s)")
    plt.ylabel("BPM")
    plt.title("Évolution de la Fréquence Cardiaque")
    plt.legend()
    plt.savefig(save_path)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    hp.plotter(wd, m, title="HeartPy Analysis")
    save_path2 = "plotsHR/HeartPy Analysis"
    plt.savefig(save_path2)

    print(f"Fréquence cardiaque moyenne estimée : {np.mean(heart_rates):.2f} BPM")
