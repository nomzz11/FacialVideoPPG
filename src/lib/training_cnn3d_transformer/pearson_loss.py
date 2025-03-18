import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

        numerator = torch.sum(pred * target, dim=-1)
        denominator = torch.sqrt(
            torch.sum(pred**2, dim=-1) * torch.sum(target**2, dim=-1)
        )

        pearson_corr = numerator / (
            denominator + 1e-8
        )  # Pour éviter la division par zéro
        loss = (
            1 - pearson_corr
        )  # Maximiser la corrélation revient à minimiser (1 - corr)
        return loss.mean()
