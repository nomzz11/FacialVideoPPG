import os, torch, json, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.lib.training.validate_model import validate_model
from src.lib.training.plot_ppg_signal import plotPpgSignal
from src.lib.training.plot_attention_maps import plot_attention_maps
from src.lib.ppg_processing.filtered_ppg import bandpass_filter


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    save_dir,
    seq_len,
    gpu=0,
    epochs=10,
    device="cuda:0",
):
    job_id = os.path.basename(save_dir)
    best_val_loss = float("inf")
    training_log = []

    if gpu == 1:
        device = "cuda:1"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Begin training... with : ", device)

    for epoch in range(epochs):
        model.train().to(device)
        train_loss = 0.0
        train_preds, train_targets = [], []

        # Training Loop
        for frames, ppg_signal in train_dataloader:
            frames, ppg_signal = frames.to(device), ppg_signal.to(device).float()
            # frames = frames.permute(0, 2, 1, 3, 4)
            optimizer.zero_grad()
            predictions, attention_maps = model(frames, seq_len=seq_len)
            predictions = predictions.squeeze()
            loss = criterion(predictions, ppg_signal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                train_preds.extend(predictions.cpu().numpy().flatten())
                train_targets.extend(ppg_signal.cpu().numpy().flatten())

        train_targets_tenseurs = torch.tensor(train_targets, dtype=torch.float32)
        train_preds_tenseurs = torch.tensor(train_preds, dtype=torch.float32)

        if (
            torch.isnan(train_targets_tenseurs).any()
            or torch.isnan(train_preds_tenseurs).any()
        ):
            print("NaN détecté dans les cibles ou les prédictions !")
        print(train_preds)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_pearson = np.float64(pearson_corrcoef(train_targets, train_preds))
        train_r2 = r2_score(train_targets, train_preds)

        # Validation
        val_loss, val_mae, val_pearson, val_r2, attention_maps_val = validate_model(
            model, val_dataloader, criterion, seq_len, device
        )

        # Storing results
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_dataloader),
            "train_mae": train_mae,
            "train_pearson": train_pearson,
            "train_r2": train_r2,
            "val_loss": val_loss / len(val_dataloader),
            "val_mae": val_mae,
            "val_pearson": val_pearson,
            "val_r2": val_r2,
        }
        training_log.append(epoch_results)

        print(
            f"[Job {job_id}] Epoch {epoch+1}/{epochs}: Train Loss: {epoch_results['train_loss']:.4f}, Val Loss: {epoch_results['val_loss']:.4f}"
        )
        print(
            f"Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}, Train R: {train_pearson:.4f}"
        )
        print(f"Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}, Val R: {val_pearson:.4f}")

        # Save the model if the validation is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # Saving training logs
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=4)

    plots_path = os.path.join(save_dir, "plots")
    os.makedirs(plots_path, exist_ok=True)
    filtered_ppg = bandpass_filter(train_preds, fs=30)
    plotPpgSignal(plots_path, train_targets, train_preds, filtered_ppg)

    plot_attention_maps(plots_path, attention_maps_val)

    print(
        f"Training finished ! Best model of Job {job_id} saved with val_loss: {best_val_loss:.4f}"
    )


def pearson_corrcoef(y_pred, y_true):
    y_pred = torch.tensor(y_pred) if isinstance(y_pred, list) else y_pred
    y_true = torch.tensor(y_true) if isinstance(y_true, list) else y_true

    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()

    return torch.sum(x * y) / (
        torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
    )
