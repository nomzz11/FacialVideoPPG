from torch.utils.data import DataLoader
import torch


def collate_fn(batch):
    """
    Regroupe les frames en séquences de 3 en s'assurant qu'elles viennent de la même vidéo.
    """
    sequences = []
    targets = []

    batch.sort(key=lambda x: x[2])  # Trier par vidéo pour garder l'ordre temporel

    current_video = None
    temp_seq = []
    temp_ppg = []

    for frame, ppg, video_name in batch:
        if frame is None or ppg is None:
            continue

        if video_name != current_video:
            temp_seq = []  # Réinitialiser si nouvelle vidéo
            temp_ppg = []
            current_video = video_name

        temp_seq.append(frame)
        temp_ppg.append(ppg)

        if len(temp_seq) == 3:  # Dès qu'on a une séquence de 3, on l'ajoute
            sequences.append(torch.stack(temp_seq))  # (3, C, H, W)
            targets.append(torch.stack(temp_ppg))  # (3,)
            temp_seq.pop(0)  # Glissement des séquences (overlapping)
            temp_ppg.pop(0)

    if len(sequences) == 0:  # Éviter d'avoir une liste vide qui casse le DataLoader
        return torch.empty(0), torch.empty(0)

    return torch.stack(sequences), torch.stack(targets)


def dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader
