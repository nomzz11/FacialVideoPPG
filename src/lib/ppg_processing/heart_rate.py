import numpy as np
import scipy.signal


def compute_heart_rate(ppg_signal, sampling_rate=30, height=None, distance=None):

    # Peak detection in the ppg signal
    peaks, _ = scipy.signal.find_peaks(ppg_signal, height=height, distance=distance)

    if len(peaks) < 2:
        return np.nan

    # Calculation of intervals between peaks (in seconds)
    peak_intervals = np.diff(peaks) / sampling_rate

    # Average the intervals to get the heart rate
    avg_rr_interval = np.mean(peak_intervals)
    heart_rate = 60 / avg_rr_interval

    return heart_rate
