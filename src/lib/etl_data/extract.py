import json, os, cv2


def extract_facial_videos(videos_paths, output_dir, transform_videos, load_frames):
    for video_path in videos_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        transform_videos(video_name, cap, output_dir, load_frames)


def extract_physiological_records(
    physiological_records_paths, output_dir, transform_records, load_records
):
    for physiological_records in physiological_records_paths:
        with open(physiological_records, "r") as f:
            transform_records(json.load(f), output_dir, load_records)
