import pandas as pd, os
import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt


def estimate_heart_rate(ppg_signal, fs=30):
    ppg_signal = np.array(ppg_signal)

    """
    filtered_extract = hp.filter_signal(
        ppg_signal, [0.7, 3.5], sample_rate=fs, order=3, filtertype="bandpass"
    )
    enhanced1 = hp.enhance_peaks(filtered_extract, iterations=2)
    """
    # Traitement du signal avec HeartPy
    try:
        wd, m = hp.process(ppg_signal, fs)
        bpm = m["bpm"]
        return wd, m, bpm

    except Exception as e:
        print("Erreur dans l'analyse du PPG :", e)
        return None


def extract_ppg_from_csv(csv_path, column_name="ppg_value"):
    """Charge un CSV et extrait les valeurs de la colonne PPG sous forme de liste."""
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(
            f"La colonne '{column_name}' n'existe pas dans le fichier CSV."
        )
    return (
        df[column_name].dropna().tolist()
    )  # Suppression des valeurs NaN si nécessaire


# Exemple d'utilisation
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(
    base_dir,
    "../refined_data/0a687dbdecde4cf1b25e00e5f513a323_2/physiological_record.csv",
)

ppg_signal = extract_ppg_from_csv(csv_dir)

print(ppg_signal[:10])  # Affiche les 10 premières valeurs

wd, m, bpm = estimate_heart_rate(ppg_signal)
hp.plotter(wd, m, title="section", figsize=(12, 6))
plt.show()
save_path2 = "section.png"
plt.savefig(save_path2)

save_path = "hr.png"

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ppg_signal, label="PPG brut", alpha=0.6)
# plt.plot(ppg_signal, label="PPG filtré", alpha=0.8)
plt.title("Signal PPG extrait")
plt.legend()
plt.savefig(save_path)

if bpm is None:
    print("L'estimation de la fréquence cardiaque a échoué.")
    pass

# Graphique du Heart Rate
plt.subplot(2, 1, 2)
plt.plot(
    bpm,
    marker="o",
    linestyle="-",
    color="red",
    label="Heart Rate (BPM)",
)
plt.xlabel("Temps (s)")
plt.ylabel("BPM")
plt.title("Évolution de la Fréquence Cardiaque")
plt.legend()
plt.tight_layout()
plt.savefig(save_path)
plt.close()
