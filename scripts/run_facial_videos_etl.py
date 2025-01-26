import os

from src.lib.etl_data import (
    ETL_data,
    extract_facial_videos,
    extract_physiological_records,
    transform_facial_video_to_frames,
    transform_physiological_records,
    load_facial_video_frame,
    load_physiological_records,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "../data")
output_dir = os.path.join(base_dir, "../refined_data")

ETL_data(
    data_dir,
    output_dir,
    extract_facial_videos,
    transform_facial_video_to_frames,
    load_facial_video_frame,
    extract_physiological_records,
    transform_physiological_records,
    load_physiological_records,
)
