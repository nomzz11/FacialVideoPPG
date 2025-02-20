import os, torch, json
from sklearn.metrics import mean_absolute_error, r2_score
from src.lib.training.validate_model import validate_model


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    save_dir,
    epochs=10,
    device="cuda",
):
    job_id = os.path.basename(save_dir)
    best_val_loss = float("inf")
    training_log = []

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Begin training...")

    for epoch in range(epochs):
        model.train().to(device)
        train_loss = 0.0
        train_preds, train_targets = [], []

        # Training Loop
        for frames, ppg_signal in train_dataloader:

            frames, ppg_signal = frames.to(device), ppg_signal.to(device).float()
            optimizer.zero_grad()
            predictions = model(frames).squeeze()
            loss = criterion(predictions, ppg_signal)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                train_preds.extend(predictions.cpu().numpy().flatten())
                train_targets.extend(ppg_signal.cpu().numpy().flatten())

        train_mae = mean_absolute_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)

        # Validation
        val_loss, val_mae, val_r2 = validate_model(
            model, val_dataloader, criterion, device
        )

        # Storing results
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_dataloader),
            "train_mae": train_mae,
            "train_r2": train_r2,
            "val_loss": val_loss / len(val_dataloader),
            "val_mae": val_mae,
            "val_r2": val_r2,
        }
        training_log.append(epoch_results)

        print(
            f"[Job {job_id}] Epoch {epoch+1}/{epochs}: Train Loss: {epoch_results['train_loss']:.4f}, Val Loss: {epoch_results['val_loss']:.4f}"
        )
        print(f"Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}")
        print(f"Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")

        # Save the model if the validation is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # Saving training logs
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=4)

    print(
        f"Training finished ! Best model of Job {job_id} saved with val_loss: {best_val_loss:.4f}"
    )
