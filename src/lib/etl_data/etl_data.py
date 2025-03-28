import os


def ETL_data(
    data_dir,
    output_dir,
    extract_videos,
    transform_videos,
    load_frames,
    extract_records,
    transform_records,
    load_records,
    options,
):
    if not os.path.isdir(data_dir):
        raise ValueError(f"Le chemin spécifié n'est pas un dossier : {data_dir}")
    files_by_type = {"videos": [], "physiological_records": []}

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            if file.endswith(".mp4"):
                files_by_type["videos"].append(file_path)
            elif file.endswith(".json"):
                files_by_type["physiological_records"].append(file_path)

    if "frames" in options or len(options) == 0:
        extract_videos(
            files_by_type["videos"], output_dir, transform_videos, load_frames
        )

    if "records" in options or len(options) == 0:
        extract_records(
            files_by_type["physiological_records"],
            output_dir,
            transform_records,
            load_records,
        )

    if ".mp4" in options[0]:
        file_path = os.path.join(data_dir, options[0])
        extract_videos(
            [x for x in files_by_type["videos"] if x == file_path],
            output_dir,
            transform_videos,
            load_frames,
        )
