import torch, numpy as np
from src.lib.pipeline_video.face_detection import detect_faces
from src.lib.pipeline_video.predict_ppg import predict_ppg
from src.lib.pipeline_video.estimate_heart_rate import estimate_heart_rate
from src.lib.training_cnn3d_transformer import r3d_transformer


def process_video(video_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = r3d_transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    faces = detect_faces(video_path)
    if not faces:
        print("Aucun visage détecté.")
        return None

    ppg_signal = predict_ppg(model, faces, device)
    result = estimate_heart_rate(ppg_signal)

    if result is None:
        return video_path, ppg_signal, None

    _, _, heart_rates = result

    return video_path, ppg_signal, np.mean(heart_rates)
