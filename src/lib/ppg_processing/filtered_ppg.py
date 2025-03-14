import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=4.0, order=3):
    signal = np.asarray(signal)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
