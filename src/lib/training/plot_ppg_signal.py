import os
import matplotlib as plt


def plotPpgSignal(dir, targets, preds):
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label="Ground Truth", color="blue", alpha=0.7)
    plt.plot(preds, label="Predictions", color="red", linestyle="dashed", alpha=0.7)
    plt.xlabel("Frame Index")
    plt.ylabel("PPG Value")
    plt.title(f"PPG Predictions vs Ground Truth")
    plt.legend()
    plt.savefig(os.path.join(dir, f"train_preds_vs_targets.png"))
    plt.close()
