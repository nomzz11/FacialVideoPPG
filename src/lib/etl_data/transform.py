import os
from .utils import find_closest_ppg


def transform_facial_video_to_frames(video_name, cap, output_dir, load_frames):
    output_frame_dir = os.path.join(output_dir, video_name)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        load_frames(frame, frame_id, output_frame_dir)
        frame_id += 1
    cap.release()


def transform_physiological_records(physiological_records, output_dir, load_records):
    physiological_records_rows = []
    current_video_name = None
    for scenario in physiological_records["scenarios"]:
        scenario_settings = scenario["scenario_settings"]
        if (
            scenario_settings["position"] == "Sitting"
            and scenario_settings["facial_movement"] == "No movement"
            and scenario_settings["talking"] == "N"
        ):
            video_name = scenario["recordings"]["RGB"]["filename"].split(".")[0]

            if video_name != current_video_name:
                # Si ce n'est pas la première vidéo, on enregistre le CSV précédent
                if current_video_name is not None:
                    load_records(
                        physiological_records_rows, current_video_name, output_dir
                    )

                # On réinitialise la liste des frames pour la nouvelle vidéo
                physiological_records_rows = []
                current_video_name = video_name

            frame_timestamps = [
                ts[0] for ts in scenario["recordings"]["RGB"]["timeseries"]
            ]

            ppg_data = scenario["recordings"]["ppg"]
            ppg_timestamps = [ts[0] for ts in ppg_data["timeseries"]]
            ppg_values = [ts[1] for ts in ppg_data["timeseries"]]

            closest_ppg_values = find_closest_ppg(
                frame_timestamps, ppg_timestamps, ppg_values
            )

            for frame_id, (frame_ts, ppg_val) in enumerate(
                zip(frame_timestamps, closest_ppg_values)
            ):
                physiological_records_rows.append(
                    {
                        "video_name": video_name,
                        "frame_name": f"{frame_id:04d}",
                        "timestamp_frame": frame_ts,
                        "ppg_value": ppg_val,
                    }
                )

        load_records(physiological_records_rows, video_name, output_dir)
