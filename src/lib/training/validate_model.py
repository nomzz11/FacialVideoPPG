import torch
from sklearn.metrics import mean_absolute_error, r2_score


def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []

    with torch.no_grad():

        for frames, ppg_values in val_dataloader:
            if frames is None or ppg_values is None:
                continue
            frames, ppg_values = frames.to(device), ppg_values.to(device).float()
            outputs, attention_maps = model(frames).squeeze()
            loss = criterion(outputs, ppg_values)
            val_loss += loss.item()

            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(ppg_values.cpu().numpy().flatten())

    # Calculation of metrics
    mae = mean_absolute_error(val_targets, val_preds)
    r2 = r2_score(val_targets, val_preds)

    return val_loss, mae, r2, attention_maps
