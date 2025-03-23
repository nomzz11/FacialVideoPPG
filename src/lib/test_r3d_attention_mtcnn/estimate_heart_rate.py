import numpy as np
import heartpy as hp


def estimate_heart_rate(ppg_signal, fs=30):
    """
    Estime la fréquence cardiaque à partir du signal PPG.
    """
    try:
        if len(ppg_signal) < fs * 2:  # Au moins 2 secondes de données
            print("Signal PPG trop court pour estimer la fréquence cardiaque")
            return None

        # Filtrage du signal pour isoler les fréquences cardiaques
        filtered_signal = hp.filter_signal(
            ppg_signal,
            [0.7, 3.5],  # Bande passante pour les fréquences cardiaques (42-210 BPM)
            sample_rate=fs,
            order=3,
            filtertype="bandpass",
        )

        # Analyse du signal avec HeartPy
        wd, m = hp.process(filtered_signal, fs)

        # Extraction du BPM moyen
        bpm = m["bpm"]

        return bpm

    except Exception as e:
        print(f"Erreur lors de l'estimation de la fréquence cardiaque: {e}")
        return None
