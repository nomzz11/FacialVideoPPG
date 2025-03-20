import numpy as np
import heartpy as hp


def estimate_heart_rate(ppg_signal, fs=30):
    ppg_signal = np.array(ppg_signal)

    try:
        filtered_extract = hp.filter_signal(
            ppg_signal, [0.7, 3.5], sample_rate=fs, order=3, filtertype="bandpass"
        )

        wd, m = hp.process(filtered_extract, fs)
        bpm = m["bpm"]
        return wd, m, bpm
    except Exception as e:
        print("Erreur dans l'analyse du PPG :", e)
        return None
