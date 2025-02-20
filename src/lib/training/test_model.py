import torch
from sklearn.metrics import mean_absolute_error, r2_score


def test_model(model, test_dataloader, criterion, device):

    model.eval()

    test_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for frames, ppg_values in test_dataloader:

            frames, ppg_values = frames.to(device), ppg_values.to(device)
            outputs = model(frames)
            loss = criterion(outputs, ppg_values)
            test_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(ppg_values.cpu().numpy().flatten())

    test_loss /= len(test_dataloader)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"Test Loss: {test_loss:.4f} - MAE: {mae:.4f} - R²: {r2:.4f}")

    return test_loss, mae, r2
