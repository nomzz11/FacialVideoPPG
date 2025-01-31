import os, pandas as pd, cv2


def load_facial_video_frame(frame, frame_name, output_frame_dir):
    if not os.path.exists(output_frame_dir):
        os.makedirs(output_frame_dir)

    frame_path = os.path.join(output_frame_dir, f"{frame_name:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    print(frame_name, os.path.basename(output_frame_dir))


def load_physiological_records(records, records_name, output_records_dir):
    output_records_path = os.path.join(
        output_records_dir, records_name, "physiological_record.csv"
    )
    df = pd.DataFrame(records)
    df = df.iloc[:-1]

    output_dir = os.path.join(output_records_dir, records_name)
    if not os.path.exists(output_dir):
        print(f"Le chemin {output_dir} n'existe pas. Opération annulée.")
        return
    df.to_csv(output_records_path, index=False)
    print(records_name, os.path.basename(output_records_dir))
