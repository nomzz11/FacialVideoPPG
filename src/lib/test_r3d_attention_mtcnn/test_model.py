import torch, numpy as np, json, os
from sklearn.metrics import mean_absolute_error, r2_score


def test_model(model, test_dataloader, criterion, save_dir, device, seq_len):
    model.to(device).eval()
    test_loss = 0.0
    test_preds, test_targets = [], []
    test_log = []
    video_predictions = {}

    with torch.no_grad():
        for frames, ppg_values, videos_name in test_dataloader:
            if frames is None or ppg_values is None:
                continue

            frames, ppg_values = frames.to(device), ppg_values.to(device).float()
            frames = frames.permute(0, 2, 1, 3, 4)

            outputs, attention_maps = model(frames, seq_len=seq_len)
            outputs = outputs.squeeze()
            loss = criterion(outputs, ppg_values)
            test_loss += loss.item()

            # Conversion en numpy
            batch_preds = outputs.cpu().numpy()
            batch_targets = ppg_values.cpu().numpy()

            # Pour les métriques globales
            if isinstance(batch_preds, np.ndarray) and batch_preds.ndim > 0:
                test_preds.extend(batch_preds.flatten())
                test_targets.extend(batch_targets.flatten())
            else:
                test_preds.append(float(batch_preds))
                test_targets.append(float(batch_targets))

            # Stockage par vidéo
            for i, video_name in enumerate(videos_name):
                if video_name not in video_predictions:
                    video_predictions[video_name] = []

                # Extraire la prédiction pour cette vidéo
                if isinstance(batch_preds, np.ndarray):
                    if batch_preds.ndim > 1:
                        # Si batch_preds a plusieurs dimensions (ex: [batch, seq])
                        pred = batch_preds[i]
                    else:
                        # Si batch_preds est 1D (ex: [batch])
                        pred = batch_preds[i] if i < len(batch_preds) else batch_preds
                else:
                    # Si c'est un scalaire
                    pred = float(batch_preds)

                # Ajouter à la liste pour cette vidéo
                if isinstance(pred, np.ndarray):
                    video_predictions[video_name].extend(pred)
                else:
                    video_predictions[video_name].append(pred)

    # Calculation of metrics
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_pearson = np.float64(pearson_corrcoef(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)

    # Storing results
    results = {
        "test_loss": test_loss / len(test_dataloader),
        "test_mae": test_mae,
        "test_pearson": test_pearson,
        "test_r2": test_r2,
    }
    test_log.append(results)

    log_path = os.path.join(save_dir, "test_log.json")
    with open(log_path, "w") as f:
        json.dump(test_log, f, indent=4)

    print(f"Test finished !!")

    return video_predictions


def pearson_corrcoef(y_pred, y_true):
    y_pred = torch.tensor(y_pred) if isinstance(y_pred, list) else y_pred
    y_true = torch.tensor(y_true) if isinstance(y_true, list) else y_true

    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()

    return torch.sum(x * y) / (
        torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
    )
