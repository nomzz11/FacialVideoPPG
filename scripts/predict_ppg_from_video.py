import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lib.pipeline_video.process_video import process_video

base_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_dir, "../data/0a687dbdecde4cf1b25e00e5f513a323_1.mp4")
model_dir = os.path.join(base_dir, "../experiments/0008/best_model.pth")

process_video(video_dir, model_dir)
