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
            outputs, attention_maps = model(frames)
            outputs = outputs.squeeze()
            loss = criterion(outputs, ppg_values)
            val_loss += loss.item()

            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(ppg_values.cpu().numpy().flatten())

    # Calculation of metrics
    mae = mean_absolute_error(val_targets, val_preds)
    pearson = pearson_corrcoef(val_targets, val_preds)
    r2 = r2_score(val_targets, val_preds)

    return val_loss, mae, pearson, r2, attention_maps


def pearson_corrcoef(y_pred, y_true):
    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()
    return torch.sum(x * y) / (
        torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
    )
