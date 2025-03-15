import os
import matplotlib.pyplot as plt


def plotPpgSignal(dir, targets, preds, filtered_preds):
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label="Ground Truth", color="blue", alpha=0.7)
    plt.plot(preds, label="Predictions", color="red", linestyle="dashed", alpha=0.7)
    plt.plot(
        filtered_preds,
        label="Predictions filtrées",
        color="green",
        linestyle="dashed",
        alpha=0.7,
    )
    plt.xlabel("Frame Index")
    plt.ylabel("PPG Value")
    plt.title("PPG Predictions vs Ground Truth")
    plt.legend()
    plt.savefig(os.path.join(dir, "train_preds_vs_targets.png"))
    plt.close()
