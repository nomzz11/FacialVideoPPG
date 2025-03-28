import numpy as np, matplotlib.pyplot as plt, os
import heartpy as hp


def estimate_heart_rate(ppg_signal, fs=30):
    ppg_signal = np.array(ppg_signal)

    try:
        filtered_extract = hp.filter_signal(
            ppg_signal, [0.7, 3.5], sample_rate=fs, order=3, filtertype="bandpass"
        )

        wd, m = hp.process(filtered_extract, fs)
        bpm = m["bpm"]

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        save_path = os.path.join(project_root, "plotsHR")
        plt.figure(figsize=(12, 6))
        hp.plotter(wd, m)
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )  # Sauvegarde en haute qualité
        plt.close()

        return wd, m, bpm
    except Exception as e:
        print("Erreur dans l'analyse du PPG :", e)
        return None
