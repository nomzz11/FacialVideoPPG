def find_closest_ppg(frame_timestamps, ppg_timestamps, ppg_values):
    frame_timestamps = np.array(frame_timestamps)
    ppg_timestamps = np.array(ppg_timestamps)
    ppg_values = np.array(ppg_values)

    closest_indices = np.abs(frame_timestamps[:, None] - ppg_timestamps).argmin(axis=1)
    closest_ppg_values = ppg_values[closest_indices]
    return closest_ppg_values
